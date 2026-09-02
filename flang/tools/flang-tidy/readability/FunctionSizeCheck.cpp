//===--- FunctionSizeCheck.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionSizeCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::readability {

using namespace parser::literals;

FunctionSizeCheck::FunctionSizeCheck(llvm::StringRef name,
                                     FlangTidyContext *context)
    : FlangTidyCheck(name, context),
      LineThreshold(Options.get("LineThreshold", DefaultLineThreshold)),
      ParameterThreshold(
          Options.get("ParameterThreshold", DefaultParameterThreshold)),
      NestingThreshold(
          Options.get("NestingThreshold", DefaultNestingThreshold)),
      inProcedure_(false), currentNestingLevel_(0), maxNestingLevel_(0) {}

void FunctionSizeCheck::storeOptions(FlangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LineThreshold", LineThreshold);
  Options.store(Opts, "ParameterThreshold", ParameterThreshold);
  Options.store(Opts, "NestingThreshold", NestingThreshold);
}

void FunctionSizeCheck::Enter(const parser::SubroutineSubprogram &program) {
  currentNestingLevel_ = -1; // start at -1 to account for the subroutine itself
  maxNestingLevel_ = 0;
  inProcedure_ = true;

  currentProcLoc_ =
      std::get<parser::Statement<parser::SubroutineStmt>>(program.t).source;

  if (ParameterThreshold) {
    const auto &subroutineStmt =
        std::get<parser::Statement<parser::SubroutineStmt>>(program.t)
            .statement;
    const auto &dummyArgs =
        std::get<std::list<parser::DummyArg>>(subroutineStmt.t);
    if ((int)dummyArgs.size() > ParameterThreshold) {
      Say(currentProcLoc_,
          "%s has %d dummy arguments, which exceeds the threshold of %d"_warn_en_US,
          currentProcLoc_, dummyArgs.size(), ParameterThreshold.value());
    }
  }

  if (LineThreshold) {
    // get the end of the subroutine
    const auto &endLoc =
        std::get<parser::Statement<parser::EndSubroutineStmt>>(program.t)
            .source;

    const auto &cookedSources =
        context()->getSemanticsContext().allCookedSources();

    // get the source position of the end location
    auto endProvenanceRange = cookedSources.GetProvenanceRange(endLoc);
    auto startProvenanceRange =
        cookedSources.GetProvenanceRange(currentProcLoc_);

    if (!endProvenanceRange || !startProvenanceRange) {
      return;
    }

    // get the source position of the end location
    auto endSourcePosition = cookedSources.allSources().GetSourcePosition(
        endProvenanceRange->start());
    auto startSourcePosition = cookedSources.allSources().GetSourcePosition(
        startProvenanceRange->start());

    if (!endSourcePosition || !startSourcePosition) {
      return;
    }

    auto lineCount = endSourcePosition->line - startSourcePosition->line;
    if (lineCount > LineThreshold) {
      Say(currentProcLoc_,
          "%s has a line count of %d, which exceeds the threshold of %d"_warn_en_US,
          currentProcLoc_, lineCount, LineThreshold.value());
    }
  }
}

void FunctionSizeCheck::Leave(const parser::SubroutineSubprogram &) {
  CheckNestingThreshold();
  inProcedure_ = false;
}

void FunctionSizeCheck::Enter(
    const parser::FunctionSubprogram &functionSubprogram) {
  currentNestingLevel_ = -1;
  maxNestingLevel_ = 0;
  inProcedure_ = true;

  currentProcLoc_ =
      std::get<parser::Statement<parser::FunctionStmt>>(functionSubprogram.t)
          .source;

  if (ParameterThreshold) {
    const auto &functionStmt =
        std::get<parser::Statement<parser::FunctionStmt>>(functionSubprogram.t)
            .statement;
    const auto &args = std::get<std::list<parser::Name>>(functionStmt.t);
    if ((int)args.size() > ParameterThreshold) {
      Say(currentProcLoc_,
          "%s has %d dummy arguments, which exceeds the threshold of %d"_warn_en_US,
          currentProcLoc_, args.size(), ParameterThreshold.value());
    }
  }

  if (LineThreshold) {
    // get the end of the subroutine
    const auto &endLoc = std::get<parser::Statement<parser::EndFunctionStmt>>(
                             functionSubprogram.t)
                             .source;

    const auto &cookedSources =
        context()->getSemanticsContext().allCookedSources();

    // get the source position of the end location
    auto endProvenanceRange = cookedSources.GetProvenanceRange(endLoc);
    auto startProvenanceRange =
        cookedSources.GetProvenanceRange(currentProcLoc_);

    if (!endProvenanceRange || !startProvenanceRange) {
      return;
    }

    // get the source position of the end location
    auto endSourcePosition = cookedSources.allSources().GetSourcePosition(
        endProvenanceRange->start());
    auto startSourcePosition = cookedSources.allSources().GetSourcePosition(
        startProvenanceRange->start());

    if (!endSourcePosition || !startSourcePosition) {
      return;
    }

    auto lineCount = endSourcePosition->line - startSourcePosition->line;
    if (lineCount > LineThreshold) {
      Say(currentProcLoc_,
          "%s has a line count of %d, which exceeds the threshold of %d"_warn_en_US,
          currentProcLoc_, lineCount, LineThreshold.value());
    }
  }
}

void FunctionSizeCheck::Leave(const parser::FunctionSubprogram &) {
  CheckNestingThreshold();
  inProcedure_ = false;
}

void FunctionSizeCheck::Enter(const parser::Block &) {
  if (inProcedure_) {
    currentNestingLevel_++;
    UpdateMaxNestingLevel();
  }
}

void FunctionSizeCheck::Leave(const parser::Block &) {
  if (inProcedure_ && currentNestingLevel_ > 0) {
    currentNestingLevel_--;
  }
}

void FunctionSizeCheck::UpdateMaxNestingLevel() {
  maxNestingLevel_ = std::max(maxNestingLevel_, currentNestingLevel_);
}

void FunctionSizeCheck::CheckNestingThreshold() {
  if (!NestingThreshold) {
    return;
  }

  if (maxNestingLevel_ > NestingThreshold) {
    Say(currentProcLoc_,
        "%s has a nesting level of %d, which exceeds the threshold of %d"_warn_en_US,
        currentProcLoc_, maxNestingLevel_, NestingThreshold.value());
  }
}

} // namespace Fortran::tidy::readability
