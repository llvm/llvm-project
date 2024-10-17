//===--- Warnings.cpp - C-Language Front-end ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command line warning options handler.
//
//===----------------------------------------------------------------------===//
//
// This file is responsible for handling all warning options. This includes
// a number of -Wfoo options and their variants, which are driven by TableGen-
// generated data, and the special cases -pedantic, -pedantic-errors, -w,
// -Werror and -Wfatal-errors.
//
// Each warning option controls any number of actual warnings.
// Given a warning option 'foo', the following are valid:
//    -Wfoo, -Wno-foo, -Werror=foo, -Wfatal-errors=foo
//
// Remark options are also handled here, analogously, except that they are much
// simpler because a remark can't be promoted to an error.
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
using namespace clang;

// EmitUnknownDiagWarning - Emit a warning and typo hint for unknown warning
// opts
static void EmitUnknownDiagWarning(DiagnosticsEngine &Diags,
                                   diag::Flavor Flavor, StringRef Prefix,
                                   StringRef Opt) {
  StringRef Suggestion = DiagnosticIDs::getNearestOption(Flavor, Opt);
  Diags.Report(diag::warn_unknown_diag_option)
      << (Flavor == diag::Flavor::WarningOrError ? 0 : 1)
      << (Prefix.str() += std::string(Opt)) << !Suggestion.empty()
      << (Prefix.str() += std::string(Suggestion));
}

namespace {
class WarningsSpecialCaseList : public llvm::SpecialCaseList {
public:
  static std::unique_ptr<WarningsSpecialCaseList>
  create(const llvm::MemoryBuffer &MB, std::string &Err) {
    auto SCL = std::make_unique<WarningsSpecialCaseList>();
    if (SCL->createInternal(&MB, Err))
      return SCL;
    return nullptr;
  }

  // Section names refer to diagnostic groups, which cover multiple individual
  // diagnostics. Expand diagnostic groups here to individual diagnostics.
  // A diagnostic can have multiple diagnostic groups associated with it, we let
  // the last section take precedence in such cases.
  void processSections(DiagnosticsEngine &Diags) {
    // Drop the default section introduced by special case list, we only support
    // exact diagnostic group names.
    Sections.erase("*");
    // Make sure we iterate sections by their line numbers.
    std::vector<std::pair<unsigned, const llvm::StringMapEntry<Section> *>>
        LineAndSection;
    for (const auto &Entry : Sections) {
      LineAndSection.emplace_back(
          Entry.second.SectionMatcher->Globs.at(Entry.first()).second, &Entry);
    }
    llvm::sort(LineAndSection);
    for (const auto &[_, Entry] : LineAndSection) {
      SmallVector<diag::kind, 256> GroupDiags;
      if (Diags.getDiagnosticIDs()->getDiagnosticsInGroup(
              clang::diag::Flavor::WarningOrError, Entry->first(),
              GroupDiags)) {
        EmitUnknownDiagWarning(Diags, clang::diag::Flavor::WarningOrError, "",
                               Entry->first());
        continue;
      }
      for (auto D : GroupDiags)
        DiagToSection[D] = &Entry->getValue();
    }
  }

  bool isDiagSuppressed(diag::kind D, llvm::StringRef FilePath) const {
    auto Section = DiagToSection.find(D);
    if (Section == DiagToSection.end())
      return false;
    auto SrcEntries = Section->second->Entries.find("src");
    if (SrcEntries == Section->second->Entries.end())
      return false;
    // Find the longest glob pattern that matches FilePath. A positive match
    // implies D should be suppressed for FilePath.
    llvm::StringRef LongestMatch;
    bool LongestWasNegative;
    for (const auto &CatIt : SrcEntries->second) {
      bool IsNegative = CatIt.first() == "emit";
      for (const auto &GlobIt : CatIt.second.Globs) {
        if (GlobIt.getKeyLength() < LongestMatch.size())
          continue;
        if (!GlobIt.second.first.match(FilePath))
          continue;
        LongestMatch = GlobIt.getKey();
        LongestWasNegative = IsNegative;
      }
    }
    return !LongestMatch.empty() && !LongestWasNegative;
  }

private:
  llvm::DenseMap<diag::kind, const Section *> DiagToSection;
};

void parseSuppressionMappings(const llvm::MemoryBuffer &MB,
                              DiagnosticsEngine &Diags) {
  std::string Err;
  auto SCL = WarningsSpecialCaseList::create(MB, Err);
  if (!SCL) {
    Diags.Report(diag::err_drv_malformed_warning_suppression_mapping)
        << MB.getBufferIdentifier() << Err;
    return;
  }
  SCL->processSections(Diags);
  Diags.setDiagSuppressionMapping(
      [SCL(std::move(SCL))](diag::kind K, llvm::StringRef Path) {
        return SCL->isDiagSuppressed(K, Path);
      });
}
} // namespace

void clang::ProcessWarningOptions(DiagnosticsEngine &Diags,
                                  const DiagnosticOptions &Opts,
                                  llvm::vfs::FileSystem &VFS,
                                  bool ReportDiags) {
  Diags.setSuppressSystemWarnings(true);  // Default to -Wno-system-headers
  Diags.setIgnoreAllWarnings(Opts.IgnoreWarnings);
  Diags.setShowOverloads(Opts.getShowOverloads());

  Diags.setElideType(Opts.ElideType);
  Diags.setPrintTemplateTree(Opts.ShowTemplateTree);
  Diags.setShowColors(Opts.ShowColors);

  // Handle -ferror-limit
  if (Opts.ErrorLimit)
    Diags.setErrorLimit(Opts.ErrorLimit);
  if (Opts.TemplateBacktraceLimit)
    Diags.setTemplateBacktraceLimit(Opts.TemplateBacktraceLimit);
  if (Opts.ConstexprBacktraceLimit)
    Diags.setConstexprBacktraceLimit(Opts.ConstexprBacktraceLimit);

  // If -pedantic or -pedantic-errors was specified, then we want to map all
  // extension diagnostics onto WARNING or ERROR unless the user has futz'd
  // around with them explicitly.
  if (Opts.PedanticErrors)
    Diags.setExtensionHandlingBehavior(diag::Severity::Error);
  else if (Opts.Pedantic)
    Diags.setExtensionHandlingBehavior(diag::Severity::Warning);
  else
    Diags.setExtensionHandlingBehavior(diag::Severity::Ignored);

  if (!Opts.SuppressionMappingsFile.empty()) {
    if (auto Buf = VFS.getBufferForFile(Opts.SuppressionMappingsFile)) {
      parseSuppressionMappings(**Buf, Diags);
    } else if (ReportDiags) {
      Diags.Report(diag::err_drv_no_such_file) << Opts.SuppressionMappingsFile;
    }
  }

  SmallVector<diag::kind, 10> _Diags;
  const IntrusiveRefCntPtr< DiagnosticIDs > DiagIDs =
    Diags.getDiagnosticIDs();
  // We parse the warning options twice.  The first pass sets diagnostic state,
  // while the second pass reports warnings/errors.  This has the effect that
  // we follow the more canonical "last option wins" paradigm when there are
  // conflicting options.
  for (unsigned Report = 0, ReportEnd = 2; Report != ReportEnd; ++Report) {
    bool SetDiagnostic = (Report == 0);

    // If we've set the diagnostic state and are not reporting diagnostics then
    // we're done.
    if (!SetDiagnostic && !ReportDiags)
      break;

    for (unsigned i = 0, e = Opts.Warnings.size(); i != e; ++i) {
      const auto Flavor = diag::Flavor::WarningOrError;
      StringRef Opt = Opts.Warnings[i];
      StringRef OrigOpt = Opts.Warnings[i];

      // Treat -Wformat=0 as an alias for -Wno-format.
      if (Opt == "format=0")
        Opt = "no-format";

      // Check to see if this warning starts with "no-", if so, this is a
      // negative form of the option.
      bool isPositive = !Opt.consume_front("no-");

      // Figure out how this option affects the warning.  If -Wfoo, map the
      // diagnostic to a warning, if -Wno-foo, map it to ignore.
      diag::Severity Mapping =
          isPositive ? diag::Severity::Warning : diag::Severity::Ignored;

      // -Wsystem-headers is a special case, not driven by the option table.  It
      // cannot be controlled with -Werror.
      if (Opt == "system-headers") {
        if (SetDiagnostic)
          Diags.setSuppressSystemWarnings(!isPositive);
        continue;
      }

      // -Weverything is a special case as well.  It implicitly enables all
      // warnings, including ones not explicitly in a warning group.
      if (Opt == "everything") {
        if (SetDiagnostic) {
          if (isPositive) {
            Diags.setEnableAllWarnings(true);
          } else {
            Diags.setEnableAllWarnings(false);
            Diags.setSeverityForAll(Flavor, diag::Severity::Ignored);
          }
        }
        continue;
      }

      // -Werror/-Wno-error is a special case, not controlled by the option
      // table. It also has the "specifier" form of -Werror=foo. GCC supports
      // the deprecated -Werror-implicit-function-declaration which is used by
      // a few projects.
      if (Opt.starts_with("error")) {
        StringRef Specifier;
        if (Opt.size() > 5) {  // Specifier must be present.
          if (Opt[5] != '=' &&
              Opt.substr(5) != "-implicit-function-declaration") {
            if (Report)
              Diags.Report(diag::warn_unknown_warning_specifier)
                << "-Werror" << ("-W" + OrigOpt.str());
            continue;
          }
          Specifier = Opt.substr(6);
        }

        if (Specifier.empty()) {
          if (SetDiagnostic)
            Diags.setWarningsAsErrors(isPositive);
          continue;
        }

        if (SetDiagnostic) {
          // Set the warning as error flag for this specifier.
          Diags.setDiagnosticGroupWarningAsError(Specifier, isPositive);
        } else if (DiagIDs->getDiagnosticsInGroup(Flavor, Specifier, _Diags)) {
          EmitUnknownDiagWarning(Diags, Flavor, "-Werror=", Specifier);
        }
        continue;
      }

      // -Wfatal-errors is yet another special case.
      if (Opt.starts_with("fatal-errors")) {
        StringRef Specifier;
        if (Opt.size() != 12) {
          if ((Opt[12] != '=' && Opt[12] != '-') || Opt.size() == 13) {
            if (Report)
              Diags.Report(diag::warn_unknown_warning_specifier)
                << "-Wfatal-errors" << ("-W" + OrigOpt.str());
            continue;
          }
          Specifier = Opt.substr(13);
        }

        if (Specifier.empty()) {
          if (SetDiagnostic)
            Diags.setErrorsAsFatal(isPositive);
          continue;
        }

        if (SetDiagnostic) {
          // Set the error as fatal flag for this specifier.
          Diags.setDiagnosticGroupErrorAsFatal(Specifier, isPositive);
        } else if (DiagIDs->getDiagnosticsInGroup(Flavor, Specifier, _Diags)) {
          EmitUnknownDiagWarning(Diags, Flavor, "-Wfatal-errors=", Specifier);
        }
        continue;
      }

      if (Report) {
        if (DiagIDs->getDiagnosticsInGroup(Flavor, Opt, _Diags))
          EmitUnknownDiagWarning(Diags, Flavor, isPositive ? "-W" : "-Wno-",
                                 Opt);
      } else {
        Diags.setSeverityForGroup(Flavor, Opt, Mapping);
      }
    }

    for (StringRef Opt : Opts.Remarks) {
      const auto Flavor = diag::Flavor::Remark;

      // Check to see if this warning starts with "no-", if so, this is a
      // negative form of the option.
      bool IsPositive = !Opt.consume_front("no-");

      auto Severity = IsPositive ? diag::Severity::Remark
                                 : diag::Severity::Ignored;

      // -Reverything sets the state of all remarks. Note that all remarks are
      // in remark groups, so we don't need a separate 'all remarks enabled'
      // flag.
      if (Opt == "everything") {
        if (SetDiagnostic)
          Diags.setSeverityForAll(Flavor, Severity);
        continue;
      }

      if (Report) {
        if (DiagIDs->getDiagnosticsInGroup(Flavor, Opt, _Diags))
          EmitUnknownDiagWarning(Diags, Flavor, IsPositive ? "-R" : "-Rno-",
                                 Opt);
      } else {
        Diags.setSeverityForGroup(Flavor, Opt,
                                  IsPositive ? diag::Severity::Remark
                                             : diag::Severity::Ignored);
      }
    }
  }
}
