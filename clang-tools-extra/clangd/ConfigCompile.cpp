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
#include "Diagnostics.h"
#include "Feature.h"
#include "TidyProvider.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Trace.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
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
  bool Trusted = false;

  llvm::Optional<llvm::Regex>
  compileRegex(const Located<std::string> &Text,
               llvm::Regex::RegexFlags Flags = llvm::Regex::NoFlags) {
    std::string Anchored = "^(" + *Text + ")$";
    llvm::Regex Result(Anchored, Flags);
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
    llvm::SmallVector<llvm::StringLiteral> ValidValues;

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
    Trusted = F.Source.Trusted;
    if (!F.Source.Directory.empty()) {
      FragmentDirectory = llvm::sys::path::convert_to_slash(F.Source.Directory);
      if (FragmentDirectory.back() != '/')
        FragmentDirectory += '/';
    }
    compile(std::move(F.If));
    compile(std::move(F.CompileFlags));
    compile(std::move(F.Index));
    compile(std::move(F.Diagnostics));
    compile(std::move(F.Completion));
    compile(std::move(F.Hover));
    compile(std::move(F.InlayHints));
  }

  void compile(Fragment::IfBlock &&F) {
    if (F.HasUnrecognizedCondition)
      Out.Conditions.push_back([&](const Params &) { return false; });

#ifdef CLANGD_PATH_CASE_INSENSITIVE
    llvm::Regex::RegexFlags Flags = llvm::Regex::IgnoreCase;
#else
    llvm::Regex::RegexFlags Flags = llvm::Regex::NoFlags;
#endif

    auto PathMatch = std::make_unique<std::vector<llvm::Regex>>();
    for (auto &Entry : F.PathMatch) {
      if (auto RE = compileRegex(Entry, Flags))
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
      if (auto RE = compileRegex(Entry, Flags))
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
    if (F.Compiler)
      Out.Apply.push_back(
          [Compiler(std::move(**F.Compiler))](const Params &, Config &C) {
            C.CompileFlags.Edits.push_back(
                [Compiler](std::vector<std::string> &Args) {
                  if (!Args.empty())
                    Args.front() = Compiler;
                });
          });

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
          // The point to insert at. Just append when `--` isn't present.
          auto It = llvm::find(Args, "--");
          Args.insert(It, Add.begin(), Add.end());
        });
      });
    }

    if (F.CompilationDatabase) {
      llvm::Optional<Config::CDBSearchSpec> Spec;
      if (**F.CompilationDatabase == "Ancestors") {
        Spec.emplace();
        Spec->Policy = Config::CDBSearchSpec::Ancestors;
      } else if (**F.CompilationDatabase == "None") {
        Spec.emplace();
        Spec->Policy = Config::CDBSearchSpec::NoCDBSearch;
      } else {
        if (auto Path =
                makeAbsolute(*F.CompilationDatabase, "CompilationDatabase",
                             llvm::sys::path::Style::native)) {
          // Drop trailing slash to put the path in canonical form.
          // Should makeAbsolute do this?
          llvm::StringRef Rel = llvm::sys::path::relative_path(*Path);
          if (!Rel.empty() && llvm::sys::path::is_separator(Rel.back()))
            Path->pop_back();

          Spec.emplace();
          Spec->Policy = Config::CDBSearchSpec::FixedDir;
          Spec->FixedCDBPath = std::move(Path);
        }
      }
      if (Spec)
        Out.Apply.push_back(
            [Spec(std::move(*Spec))](const Params &, Config &C) {
              C.CompileFlags.CDBSearch = Spec;
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
    if (External.Server && !Trusted) {
      diag(Error,
           "Remote index may not be specified by untrusted configuration. "
           "Copy this into user config to use it.",
           External.Server->Range);
      return;
    }
#ifndef CLANGD_ENABLE_REMOTE
    if (External.Server) {
      elog("Clangd isn't compiled with remote index support, ignoring Server: "
           "{0}",
           *External.Server);
      External.Server.reset();
    }
#endif
    // Make sure exactly one of the Sources is set.
    unsigned SourceCount = External.File.hasValue() +
                           External.Server.hasValue() + *External.IsNone;
    if (SourceCount != 1) {
      diag(Error, "Exactly one of File, Server or None must be set.",
           BlockRange);
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
    } else {
      assert(*External.IsNone);
      Spec.Kind = Config::ExternalIndexSpec::None;
    }
    if (Spec.Kind != Config::ExternalIndexSpec::None) {
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
    }
    Out.Apply.push_back([Spec(std::move(Spec))](const Params &P, Config &C) {
      if (Spec.Kind == Config::ExternalIndexSpec::None) {
        C.Index.External = Spec;
        return;
      }
      if (P.Path.empty() || !pathStartsWith(Spec.MountPoint, P.Path,
                                            llvm::sys::path::Style::posix))
        return;
      C.Index.External = Spec;
      // Disable background indexing for the files under the mountpoint.
      // Note that this will overwrite statements in any previous fragments
      // (including the current one).
      C.Index.Background = Config::BackgroundPolicy::Skip;
    });
  }

  void compile(Fragment::DiagnosticsBlock &&F) {
    std::vector<std::string> Normalized;
    for (const auto &Suppressed : F.Suppress) {
      if (*Suppressed == "*") {
        Out.Apply.push_back([&](const Params &, Config &C) {
          C.Diagnostics.SuppressAll = true;
          C.Diagnostics.Suppress.clear();
        });
        return;
      }
      Normalized.push_back(normalizeSuppressedCode(*Suppressed).str());
    }
    if (!Normalized.empty())
      Out.Apply.push_back(
          [Normalized(std::move(Normalized))](const Params &, Config &C) {
            if (C.Diagnostics.SuppressAll)
              return;
            for (llvm::StringRef N : Normalized)
              C.Diagnostics.Suppress.insert(N);
          });

    if (F.UnusedIncludes)
      if (auto Val = compileEnum<Config::UnusedIncludesPolicy>(
                         "UnusedIncludes", **F.UnusedIncludes)
                         .map("Strict", Config::UnusedIncludesPolicy::Strict)
                         .map("None", Config::UnusedIncludesPolicy::None)
                         .value())
        Out.Apply.push_back([Val](const Params &, Config &C) {
          C.Diagnostics.UnusedIncludes = *Val;
        });

    compile(std::move(F.ClangTidy));
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
    StringRef Str = StringRef(*Arg).trim();
    // Don't support negating here, its handled if the item is in the Add or
    // Remove list.
    if (Str.startswith("-") || Str.contains(',')) {
      diag(Error, "Invalid clang-tidy check name", Arg.Range);
      return;
    }
    if (!Str.contains('*') && !isRegisteredTidyCheck(Str)) {
      diag(Warning,
           llvm::formatv("clang-tidy check '{0}' was not found", Str).str(),
           Arg.Range);
      return;
    }
    CurSpec += ',';
    if (!IsPositive)
      CurSpec += '-';
    CurSpec += Str;
  }

  void compile(Fragment::DiagnosticsBlock::ClangTidyBlock &&F) {
    std::string Checks;
    for (auto &CheckGlob : F.Add)
      appendTidyCheckSpec(Checks, CheckGlob, true);

    for (auto &CheckGlob : F.Remove)
      appendTidyCheckSpec(Checks, CheckGlob, false);

    if (!Checks.empty())
      Out.Apply.push_back(
          [Checks = std::move(Checks)](const Params &, Config &C) {
            C.Diagnostics.ClangTidy.Checks.append(
                Checks,
                C.Diagnostics.ClangTidy.Checks.empty() ? /*skip comma*/ 1 : 0,
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
              C.Diagnostics.ClangTidy.CheckOptions.insert_or_assign(
                  StringPair.first, StringPair.second);
          });
    }
  }

  void compile(Fragment::CompletionBlock &&F) {
    if (F.AllScopes) {
      Out.Apply.push_back(
          [AllScopes(**F.AllScopes)](const Params &, Config &C) {
            C.Completion.AllScopes = AllScopes;
          });
    }
  }

  void compile(Fragment::HoverBlock &&F) {
    if (F.ShowAKA) {
      Out.Apply.push_back([ShowAKA(**F.ShowAKA)](const Params &, Config &C) {
        C.Hover.ShowAKA = ShowAKA;
      });
    }
  }

  void compile(Fragment::InlayHintsBlock &&F) {
    if (F.Enabled)
      Out.Apply.push_back([Value(**F.Enabled)](const Params &, Config &C) {
        C.InlayHints.Enabled = Value;
      });
    if (F.ParameterNames)
      Out.Apply.push_back(
          [Value(**F.ParameterNames)](const Params &, Config &C) {
            C.InlayHints.Parameters = Value;
          });
    if (F.DeducedTypes)
      Out.Apply.push_back([Value(**F.DeducedTypes)](const Params &, Config &C) {
        C.InlayHints.DeducedTypes = Value;
      });
    if (F.Designators)
      Out.Apply.push_back([Value(**F.Designators)](const Params &, Config &C) {
        C.InlayHints.Designators = Value;
      });
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
  vlog("Config fragment: compiling {0}:{1} -> {2} (trusted={3})", ConfigFile,
       LineCol.first, Result.get(), Source.Trusted);

  FragmentCompiler{*Result, D, Source.Manager.get()}.compile(std::move(*this));
  // Return as cheaply-copyable wrapper.
  return [Result(std::move(Result))](const Params &P, Config &C) {
    return (*Result)(P, C);
  };
}

} // namespace config
} // namespace clangd
} // namespace clang
