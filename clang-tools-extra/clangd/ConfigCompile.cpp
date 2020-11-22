//===--- ConfigCompile.cpp - Translating Fragments into Config ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fragments are applied to Configs in two steps:
//
// 1. (When the fragment is first loaded)
//    FragmentCompiler::compile() traverses the Fragment and creates
//    function objects that know how to apply the configuration.
// 2. (Every time a config is required)
//    CompiledFragment() executes these functions to populate the Config.
//
// Work could be split between these steps in different ways. We try to
// do as much work as possible in the first step. For example, regexes are
// compiled in stage 1 and captured by the apply function. This is because:
//
//  - it's more efficient, as the work done in stage 1 must only be done once
//  - problems can be reported in stage 1, in stage 2 we must silently recover
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "Config.h"
#include "ConfigFragment.h"
#include "ConfigProvider.h"
#include "Features.inc"
#include "support/Logger.h"
#include "support/Trace.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

namespace clang {
namespace clangd {
namespace config {
namespace {

// Returns an empty stringref if Path is not under FragmentDir. Returns Path
// as-is when FragmentDir is empty.
llvm::StringRef configRelative(llvm::StringRef Path,
                               llvm::StringRef FragmentDir) {
  if (FragmentDir.empty())
    return Path;
  if (!Path.consume_front(FragmentDir))
    return llvm::StringRef();
  return Path.empty() ? "." : Path;
}

struct CompiledFragmentImpl {
  // The independent conditions to check before using settings from this config.
  // The following fragment has *two* conditions:
  //   If: { Platform: [mac, linux], PathMatch: foo/.* }
  // All of them must be satisfied: the platform and path conditions are ANDed.
  // The OR logic for the platform condition is implemented inside the function.
  std::vector<llvm::unique_function<bool(const Params &) const>> Conditions;
  // Mutations that this fragment will apply to the configuration.
  // These are invoked only if the conditions are satisfied.
  std::vector<llvm::unique_function<void(const Params &, Config &) const>>
      Apply;

  bool operator()(const Params &P, Config &C) const {
    for (const auto &C : Conditions) {
      if (!C(P)) {
        dlog("Config fragment {0}: condition not met", this);
        return false;
      }
    }
    dlog("Config fragment {0}: applying {1} rules", this, Apply.size());
    for (const auto &A : Apply)
      A(P, C);
    return true;
  }
};

// Wrapper around condition compile() functions to reduce arg-passing.
struct FragmentCompiler {
  FragmentCompiler(CompiledFragmentImpl &Out, DiagnosticCallback D,
                   llvm::SourceMgr *SM)
      : Out(Out), Diagnostic(D), SourceMgr(SM) {}
  CompiledFragmentImpl &Out;
  DiagnosticCallback Diagnostic;
  llvm::SourceMgr *SourceMgr;
  // Normalized Fragment::SourceInfo::Directory.
  std::string FragmentDirectory;

  llvm::Optional<llvm::Regex> compileRegex(const Located<std::string> &Text) {
    std::string Anchored = "^(" + *Text + ")$";
    llvm::Regex Result(Anchored);
    std::string RegexError;
    if (!Result.isValid(RegexError)) {
      diag(Error, "Invalid regex " + Anchored + ": " + RegexError, Text.Range);
      return llvm::None;
    }
    return Result;
  }

  llvm::Optional<std::string> makeAbsolute(Located<std::string> Path,
                                           llvm::StringLiteral Description,
                                           llvm::sys::path::Style Style) {
    if (llvm::sys::path::is_absolute(*Path))
      return *Path;
    if (FragmentDirectory.empty()) {
      diag(Error,
           llvm::formatv(
               "{0} must be an absolute path, because this fragment is not "
               "associated with any directory.",
               Description)
               .str(),
           Path.Range);
      return llvm::None;
    }
    llvm::SmallString<256> AbsPath = llvm::StringRef(*Path);
    llvm::sys::fs::make_absolute(FragmentDirectory, AbsPath);
    llvm::sys::path::native(AbsPath, Style);
    return AbsPath.str().str();
  }

  // Helper with similar API to StringSwitch, for parsing enum values.
  template <typename T> class EnumSwitch {
    FragmentCompiler &Outer;
    llvm::StringRef EnumName;
    const Located<std::string> &Input;
    llvm::Optional<T> Result;
    llvm::SmallVector<llvm::StringLiteral, 8> ValidValues;

  public:
    EnumSwitch(llvm::StringRef EnumName, const Located<std::string> &In,
               FragmentCompiler &Outer)
        : Outer(Outer), EnumName(EnumName), Input(In) {}

    EnumSwitch &map(llvm::StringLiteral Name, T Value) {
      assert(!llvm::is_contained(ValidValues, Name) && "Duplicate value!");
      ValidValues.push_back(Name);
      if (!Result && *Input == Name)
        Result = Value;
      return *this;
    }

    llvm::Optional<T> value() {
      if (!Result)
        Outer.diag(
            Warning,
            llvm::formatv("Invalid {0} value '{1}'. Valid values are {2}.",
                          EnumName, *Input, llvm::join(ValidValues, ", "))
                .str(),
            Input.Range);
      return Result;
    };
  };

  // Attempt to parse a specified string into an enum.
  // Yields llvm::None and produces a diagnostic on failure.
  //
  // Optional<T> Value = compileEnum<En>("Foo", Frag.Foo)
  //    .map("Foo", Enum::Foo)
  //    .map("Bar", Enum::Bar)
  //    .value();
  template <typename T>
  EnumSwitch<T> compileEnum(llvm::StringRef EnumName,
                            const Located<std::string> &In) {
    return EnumSwitch<T>(EnumName, In, *this);
  }

  void compile(Fragment &&F) {
    if (!F.Source.Directory.empty()) {
      FragmentDirectory = llvm::sys::path::convert_to_slash(F.Source.Directory);
      if (FragmentDirectory.back() != '/')
        FragmentDirectory += '/';
    }
    compile(std::move(F.If));
    compile(std::move(F.CompileFlags));
    compile(std::move(F.Index));
    compile(std::move(F.ClangTidy));
  }

  void compile(Fragment::IfBlock &&F) {
    if (F.HasUnrecognizedCondition)
      Out.Conditions.push_back([&](const Params &) { return false; });

    auto PathMatch = std::make_unique<std::vector<llvm::Regex>>();
    for (auto &Entry : F.PathMatch) {
      if (auto RE = compileRegex(Entry))
        PathMatch->push_back(std::move(*RE));
    }
    if (!PathMatch->empty()) {
      Out.Conditions.push_back(
          [PathMatch(std::move(PathMatch)),
           FragmentDir(FragmentDirectory)](const Params &P) {
            if (P.Path.empty())
              return false;
            llvm::StringRef Path = configRelative(P.Path, FragmentDir);
            // Ignore the file if it is not nested under Fragment.
            if (Path.empty())
              return false;
            return llvm::any_of(*PathMatch, [&](const llvm::Regex &RE) {
              return RE.match(Path);
            });
          });
    }

    auto PathExclude = std::make_unique<std::vector<llvm::Regex>>();
    for (auto &Entry : F.PathExclude) {
      if (auto RE = compileRegex(Entry))
        PathExclude->push_back(std::move(*RE));
    }
    if (!PathExclude->empty()) {
      Out.Conditions.push_back(
          [PathExclude(std::move(PathExclude)),
           FragmentDir(FragmentDirectory)](const Params &P) {
            if (P.Path.empty())
              return false;
            llvm::StringRef Path = configRelative(P.Path, FragmentDir);
            // Ignore the file if it is not nested under Fragment.
            if (Path.empty())
              return true;
            return llvm::none_of(*PathExclude, [&](const llvm::Regex &RE) {
              return RE.match(Path);
            });
          });
    }
  }

  void compile(Fragment::CompileFlagsBlock &&F) {
    if (!F.Remove.empty()) {
      auto Remove = std::make_shared<ArgStripper>();
      for (auto &A : F.Remove)
        Remove->strip(*A);
      Out.Apply.push_back([Remove(std::shared_ptr<const ArgStripper>(
                              std::move(Remove)))](const Params &, Config &C) {
        C.CompileFlags.Edits.push_back(
            [Remove](std::vector<std::string> &Args) {
              Remove->process(Args);
            });
      });
    }

    if (!F.Add.empty()) {
      std::vector<std::string> Add;
      for (auto &A : F.Add)
        Add.push_back(std::move(*A));
      Out.Apply.push_back([Add(std::move(Add))](const Params &, Config &C) {
        C.CompileFlags.Edits.push_back([Add](std::vector<std::string> &Args) {
          Args.insert(Args.end(), Add.begin(), Add.end());
        });
      });
    }
  }

  void compile(Fragment::IndexBlock &&F) {
    if (F.Background) {
      if (auto Val = compileEnum<Config::BackgroundPolicy>("Background",
                                                           **F.Background)
                         .map("Build", Config::BackgroundPolicy::Build)
                         .map("Skip", Config::BackgroundPolicy::Skip)
                         .value())
        Out.Apply.push_back(
            [Val](const Params &, Config &C) { C.Index.Background = *Val; });
    }
    if (F.External)
      compile(std::move(**F.External), F.External->Range);
  }

  void compile(Fragment::IndexBlock::ExternalBlock &&External,
               llvm::SMRange BlockRange) {
#ifndef CLANGD_ENABLE_REMOTE
    if (External.Server) {
      diag(Error, "Clangd isn't compiled with remote index support, ignoring "
                  "Server." External.Server->Range);
      External.Server.reset();
    }
#endif
    // Make sure exactly one of the Sources is set.
    unsigned SourceCount =
        External.File.hasValue() + External.Server.hasValue();
    if (SourceCount != 1) {
      diag(Error, "Exactly one of File or Server must be set.", BlockRange);
      return;
    }
    Config::ExternalIndexSpec Spec;
    if (External.Server) {
      Spec.Kind = Config::ExternalIndexSpec::Server;
      Spec.Location = std::move(**External.Server);
    } else if (External.File) {
      Spec.Kind = Config::ExternalIndexSpec::File;
      auto AbsPath = makeAbsolute(std::move(*External.File), "File",
                                  llvm::sys::path::Style::native);
      if (!AbsPath)
        return;
      Spec.Location = std::move(*AbsPath);
    }
    // Make sure MountPoint is an absolute path with forward slashes.
    if (!External.MountPoint)
      External.MountPoint.emplace(FragmentDirectory);
    if ((**External.MountPoint).empty()) {
      diag(Error, "A mountpoint is required.", BlockRange);
      return;
    }
    auto AbsPath = makeAbsolute(std::move(*External.MountPoint), "MountPoint",
                                llvm::sys::path::Style::posix);
    if (!AbsPath)
      return;
    Spec.MountPoint = std::move(*AbsPath);
    Out.Apply.push_back([Spec(std::move(Spec))](const Params &P, Config &C) {
      if (!P.Path.startswith(Spec.MountPoint))
        return;
      C.Index.External = Spec;
      // Disable background indexing for the files under the mountpoint.
      // Note that this will overwrite statements in any previous fragments
      // (including the current one).
      C.Index.Background = Config::BackgroundPolicy::Skip;
    });
  }

  void compile(Fragment::StyleBlock &&F) {
    if (!F.FullyQualifiedNamespaces.empty()) {
      std::vector<std::string> FullyQualifiedNamespaces;
      for (auto &N : F.FullyQualifiedNamespaces) {
        // Normalize the data by dropping both leading and trailing ::
        StringRef Namespace(*N);
        Namespace.consume_front("::");
        Namespace.consume_back("::");
        FullyQualifiedNamespaces.push_back(Namespace.str());
      }
      Out.Apply.push_back([FullyQualifiedNamespaces(
                              std::move(FullyQualifiedNamespaces))](
                              const Params &, Config &C) {
        C.Style.FullyQualifiedNamespaces.insert(
            C.Style.FullyQualifiedNamespaces.begin(),
            FullyQualifiedNamespaces.begin(), FullyQualifiedNamespaces.end());
      });
    }
  }

  void appendTidyCheckSpec(std::string &CurSpec,
                           const Located<std::string> &Arg, bool IsPositive) {
    StringRef Str = *Arg;
    // Don't support negating here, its handled if the item is in the Add or
    // Remove list.
    if (Str.startswith("-") || Str.contains(',')) {
      diag(Error, "Invalid clang-tidy check name", Arg.Range);
      return;
    }
    CurSpec += ',';
    if (!IsPositive)
      CurSpec += '-';
    CurSpec += Str;
  }

  void compile(Fragment::ClangTidyBlock &&F) {
    std::string Checks;
    for (auto &CheckGlob : F.Add)
      appendTidyCheckSpec(Checks, CheckGlob, true);

    for (auto &CheckGlob : F.Remove)
      appendTidyCheckSpec(Checks, CheckGlob, false);

    if (!Checks.empty())
      Out.Apply.push_back(
          [Checks = std::move(Checks)](const Params &, Config &C) {
            C.ClangTidy.Checks.append(
                Checks, C.ClangTidy.Checks.empty() ? /*skip comma*/ 1 : 0,
                std::string::npos);
          });
    if (!F.CheckOptions.empty()) {
      std::vector<std::pair<std::string, std::string>> CheckOptions;
      for (auto &Opt : F.CheckOptions)
        CheckOptions.emplace_back(std::move(*Opt.first),
                                  std::move(*Opt.second));
      Out.Apply.push_back(
          [CheckOptions = std::move(CheckOptions)](const Params &, Config &C) {
            for (auto &StringPair : CheckOptions)
              C.ClangTidy.CheckOptions.insert_or_assign(StringPair.first,
                                                        StringPair.second);
          });
    }
  }

  constexpr static llvm::SourceMgr::DiagKind Error = llvm::SourceMgr::DK_Error;
  constexpr static llvm::SourceMgr::DiagKind Warning =
      llvm::SourceMgr::DK_Warning;
  void diag(llvm::SourceMgr::DiagKind Kind, llvm::StringRef Message,
            llvm::SMRange Range) {
    if (Range.isValid() && SourceMgr != nullptr)
      Diagnostic(SourceMgr->GetMessage(Range.Start, Kind, Message, Range));
    else
      Diagnostic(llvm::SMDiagnostic("", Kind, Message));
  }
};

} // namespace

CompiledFragment Fragment::compile(DiagnosticCallback D) && {
  llvm::StringRef ConfigFile = "<unknown>";
  std::pair<unsigned, unsigned> LineCol = {0, 0};
  if (auto *SM = Source.Manager.get()) {
    unsigned BufID = SM->getMainFileID();
    LineCol = SM->getLineAndColumn(Source.Location, BufID);
    ConfigFile = SM->getBufferInfo(BufID).Buffer->getBufferIdentifier();
  }
  trace::Span Tracer("ConfigCompile");
  SPAN_ATTACH(Tracer, "ConfigFile", ConfigFile);
  auto Result = std::make_shared<CompiledFragmentImpl>();
  vlog("Config fragment: compiling {0}:{1} -> {2}", ConfigFile, LineCol.first,
       Result.get());

  FragmentCompiler{*Result, D, Source.Manager.get()}.compile(std::move(*this));
  // Return as cheaply-copyable wrapper.
  return [Result(std::move(Result))](const Params &P, Config &C) {
    return (*Result)(P, C);
  };
}

} // namespace config
} // namespace clangd
} // namespace clang
