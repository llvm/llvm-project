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

#include "Config.h"
#include "ConfigFragment.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace clang {
namespace clangd {
namespace config {
namespace {

struct CompiledFragmentImpl {
  // The independent conditions to check before using settings from this config.
  // The following fragment has *two* conditions:
  //   If: { Platform: [mac, linux], PathMatch: foo/.* }
  // All of them must be satisfied: the platform and path conditions are ANDed.
  // The OR logic for the platform condition is implemented inside the function.
  std::vector<llvm::unique_function<bool(const Params &) const>> Conditions;
  // Mutations that this fragment will apply to the configuration.
  // These are invoked only if the conditions are satisfied.
  std::vector<llvm::unique_function<void(Config &) const>> Apply;

  bool operator()(const Params &P, Config &C) const {
    for (const auto &C : Conditions) {
      if (!C(P)) {
        dlog("Config fragment {0}: condition not met", this);
        return false;
      }
    }
    dlog("Config fragment {0}: applying {1} rules", this, Apply.size());
    for (const auto &A : Apply)
      A(C);
    return true;
  }
};

// Wrapper around condition compile() functions to reduce arg-passing.
struct FragmentCompiler {
  CompiledFragmentImpl &Out;
  DiagnosticCallback Diagnostic;
  llvm::SourceMgr *SourceMgr;

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

  void compile(Fragment &&F) {
    compile(std::move(F.If));
    compile(std::move(F.CompileFlags));
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
          [PathMatch(std::move(PathMatch))](const Params &P) {
            if (P.Path.empty())
              return false;
            return llvm::any_of(*PathMatch, [&](const llvm::Regex &RE) {
              return RE.match(P.Path);
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
          [PathExclude(std::move(PathExclude))](const Params &P) {
            if (P.Path.empty())
              return false;
            return llvm::none_of(*PathExclude, [&](const llvm::Regex &RE) {
              return RE.match(P.Path);
            });
          });
    }
  }

  void compile(Fragment::CompileFlagsBlock &&F) {
    if (!F.Add.empty()) {
      std::vector<std::string> Add;
      for (auto &A : F.Add)
        Add.push_back(std::move(*A));
      Out.Apply.push_back([Add(std::move(Add))](Config &C) {
        C.CompileFlags.Edits.push_back([Add](std::vector<std::string> &Args) {
          Args.insert(Args.end(), Add.begin(), Add.end());
        });
      });
    }
  }

  constexpr static llvm::SourceMgr::DiagKind Error = llvm::SourceMgr::DK_Error;
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
