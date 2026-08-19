//===- StaticLibraryCreateCLI.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements the `static-library create` CLI action. The class contains
//  no cl::opt globals of its own: the driver in SSAFLinker.cpp owns all
//  flag definitions and hands values in via the Config struct passed to
//  run().
//
//===----------------------------------------------------------------------===//

#include "StaticLibraryCreateCLI.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <memory>

using namespace llvm;
using namespace clang::ssaf;

namespace path = llvm::sys::path;

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace {

constexpr const char *ReadingSummary = "Reading TU summary '{0}'";

constexpr const char *TargetTripleMismatch =
    "target triple '{0}' from TU summary '{1}' does not match expected "
    "triple '{2}'";

constexpr const char *NoInputs =
    "no input TU summaries: at least one input is required";

constexpr const char *DuplicateMember =
    "duplicate TU summary member with namespace {0}";

constexpr const char *InvalidTargetTriple =
    "invalid --target-triple '{0}': unrecognized {1}";

BuildNamespace makeStaticLibraryNamespace(llvm::StringRef Name) {
  return BuildNamespace(BuildNamespaceKind::StaticLibrary, Name);
}

} // namespace

//===----------------------------------------------------------------------===//
// StaticLibraryCreateCLI
//===----------------------------------------------------------------------===//

void StaticLibraryCreateCLI::run(llvm::TimerGroup &TG, const Config &InCfg) {
  Cfg = InCfg;

  info(0, "Creating StaticLibrary started.");

  {
    info(1, "Validating input.");
    validate(TG);
  }

  info(1, "Bundling StaticLibrary objects.");
  StaticLibrary Result = bundle(TG);
  write(TG, Result);

  info(0, "Creating StaticLibrary finished.");
}

void StaticLibraryCreateCLI::validate(llvm::TimerGroup &TG) {
  llvm::Timer TValidate("validate", "Validate Input", TG);
  llvm::TimeRegion _(Cfg.Time ? &TValidate : nullptr);

  OutputFile = FormatFile::fromOutputPath(Cfg.OutputPath);
  info(2, "Validated output path '{0}'.", OutputFile.Path);

  if (Cfg.InputPaths.empty()) {
    fail(NoInputs);
  }
  for (const auto &InputPath : Cfg.InputPaths) {
    InputFiles.push_back(FormatFile::fromInputPath(InputPath));
  }
  info(2, "Validated {0} input summary paths.", InputFiles.size());

  NamespaceName = Cfg.Namespace.empty() ? path::stem(OutputFile.Path).str()
                                        : Cfg.Namespace.str();
  info(2, "Namespace name: '{0}'.", NamespaceName);

  if (!Cfg.TargetTriple.empty()) {
    // Enforce that the supplied triple names a recognized architecture, vendor,
    // and OS up front so obviously-malformed values fail here with a targeted
    // message rather than surfacing later as a triple mismatch against an
    // input's real triple.
    llvm::Triple T(Cfg.TargetTriple);
    if (T.getArch() == llvm::Triple::UnknownArch) {
      fail(InvalidTargetTriple, Cfg.TargetTriple, "architecture");
    }
    if (T.getVendor() == llvm::Triple::UnknownVendor) {
      fail(InvalidTargetTriple, Cfg.TargetTriple, "vendor");
    }
    if (T.getOS() == llvm::Triple::UnknownOS) {
      fail(InvalidTargetTriple, Cfg.TargetTriple, "OS");
    }
    ResolvedTriple = std::move(T);
    info(2, "Explicit target triple: '{0}'.",
         llvm::Triple::normalize(ResolvedTriple->str()));
  }
}

StaticLibrary StaticLibraryCreateCLI::bundle(llvm::TimerGroup &TG) {
  llvm::Timer TRead("read", "Read Summaries", TG);
  llvm::Timer TAssemble("assemble", "Assemble StaticLibrary", TG);

  // If the target triple came from --target-triple, we can build the
  // StaticLibrary upfront; otherwise its triple is the first member's
  // and construction has to wait until we've read that member.
  std::optional<StaticLibrary> Result;
  if (ResolvedTriple) {
    Result.emplace(*ResolvedTriple, makeStaticLibraryNamespace(NamespaceName));
  }

  info(2, "Bundling members.");

  for (auto [Index, InputFile] : llvm::enumerate(InputFiles)) {
    std::unique_ptr<TUSummaryEncoding> Member;

    {
      info(3, "[{0}/{1}] Reading '{2}'.", (Index + 1), InputFiles.size(),
           InputFile.Path);
      llvm::TimeRegion _(Cfg.Time ? &TRead : nullptr);

      auto ExpectedSummary =
          InputFile.Format->readTUSummaryEncoding(InputFile.Path);
      if (!ExpectedSummary) {
        fail(ErrorBuilder::wrap(ExpectedSummary.takeError())
                 .context(ReadingSummary, InputFile.Path)
                 .build());
      }
      Member = std::make_unique<TUSummaryEncoding>(std::move(*ExpectedSummary));
    }

    // Resolve (or verify) the target triple and, if this is the first
    // input under triple inference, construct the StaticLibrary now.
    const llvm::Triple &MemberTriple = Member->getTargetTriple();
    if (!Result) {
      Result.emplace(MemberTriple, makeStaticLibraryNamespace(NamespaceName));
    } else if (MemberTriple != Result->TargetTriple) {
      fail(TargetTripleMismatch, llvm::Triple::normalize(MemberTriple.str()),
           InputFile.Path, llvm::Triple::normalize(Result->TargetTriple.str()));
    }

    {
      info(3, "[{0}/{1}] Assembling '{2}'.", (Index + 1), InputFiles.size(),
           InputFile.Path);
      llvm::TimeRegion _(Cfg.Time ? &TAssemble : nullptr);

      auto MemberNamespace = Member->TUNamespace;
      auto [It, Inserted] = Result->Members.insert(std::move(Member));
      if (!Inserted) {
        fail(DuplicateMember, MemberNamespace);
      }
    }
  }

  info(2, "Target triple: '{0}'.",
       llvm::Triple::normalize(Result->TargetTriple.str()));

  return std::move(*Result);
}

void StaticLibraryCreateCLI::write(llvm::TimerGroup &TG,
                                   const StaticLibrary &Result) {
  info(2, "Writing StaticLibrary to '{0}'.", OutputFile.Path);

  llvm::Timer TWrite("write", "Write StaticLibrary", TG);
  llvm::TimeRegion _(Cfg.Time ? &TWrite : nullptr);

  if (auto Err =
          OutputFile.Format->writeStaticLibrary(Result, OutputFile.Path)) {
    fail(std::move(Err));
  }
}
