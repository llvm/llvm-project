//===--- ParserActions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/ParserActions.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::frontend {

parser::AllCookedSources &getAllCooked(CompilerInstance &ci) {
  return ci.getParsing().allCooked();
}

void parseAndLowerTree(CompilerInstance &ci, lower::LoweringBridge &lb) {
  parser::Program &parseTree{*ci.getParsing().parseTree()};
  lb.lower(parseTree, ci.getSemanticsContext());
}

void dumpTree(CompilerInstance &ci) {
  auto &parseTree{ci.getParsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  parser::DumpTree(llvm::outs(), parseTree, &ci.getInvocation().getAsFortran());
}

void dumpProvenance(CompilerInstance &ci) {
  ci.getParsing().DumpProvenance(llvm::outs());
}

void dumpPreFIRTree(CompilerInstance &ci) {
  auto &parseTree{*ci.getParsing().parseTree()};

  if (auto ast{lower::createPFT(parseTree, ci.getSemanticsContext())}) {
    lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.getDiagnostics().Report(diagID);
  }
}

void formatOrDumpPrescanner(std::string &buf,
                            llvm::raw_string_ostream &outForPP,
                            CompilerInstance &ci) {
  if (ci.getInvocation().getPreprocessorOpts().showMacros) {
    ci.getParsing().EmitPreprocessorMacros(outForPP);
  } else if (ci.getInvocation().getPreprocessorOpts().noReformat) {
    ci.getParsing().DumpCookedChars(outForPP);
  } else {
    ci.getParsing().EmitPreprocessedSource(
        outForPP, !ci.getInvocation().getPreprocessorOpts().noLineDirectives);
  }

  // Print getDiagnostics from the prescanner
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());
}

struct MeasurementVisitor {
  template <typename A>
  bool Pre(const A &) {
    return true;
  }
  template <typename A>
  void Post(const A &) {
    ++objects;
    bytes += sizeof(A);
  }
  size_t objects{0}, bytes{0};
};

void debugMeasureParseTree(CompilerInstance &ci, llvm::StringRef filename) {
  // Parse. In case of failure, report and return.
  ci.getParsing().Parse(llvm::outs());

  if ((ci.getParsing().parseTree().has_value() &&
       !ci.getParsing().consumedWholeFile()) ||
      (!ci.getParsing().messages().empty() &&
       (ci.getInvocation().getWarnAsErr() ||
        ci.getParsing().messages().AnyFatalError()))) {
    unsigned diagID = ci.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.getDiagnostics().Report(diagID) << filename;

    ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());
    return;
  }

  // Report the getDiagnostics from parsing
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  auto &parseTree{ci.getParsing().parseTree()};
  MeasurementVisitor visitor;
  parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void debugUnparseNoSema(CompilerInstance &ci, llvm::raw_ostream &out) {
  auto &invoc = ci.getInvocation();
  auto &parseTree{ci.getParsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(out, *parseTree,
          /*encoding=*/parser::Encoding::UTF_8,
          /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
          /*preStatement=*/nullptr,
          invoc.getUseAnalyzedObjectsForUnparse() ? &invoc.getAsFortran()
                                                  : nullptr);
}

void debugUnparseWithSymbols(CompilerInstance &ci) {
  auto &parseTree{*ci.getParsing().parseTree()};

  semantics::UnparseWithSymbols(llvm::outs(), parseTree,
                                /*encoding=*/parser::Encoding::UTF_8);
}

void debugUnparseWithModules(CompilerInstance &ci) {
  auto &parseTree{*ci.getParsing().parseTree()};
  semantics::UnparseWithModules(llvm::outs(), ci.getSemantics().context(),
                                parseTree,
                                /*encoding=*/parser::Encoding::UTF_8);
}

void debugDumpParsingLog(CompilerInstance &ci) {
  ci.getParsing().Parse(llvm::errs());
  ci.getParsing().DumpParsingLog(llvm::outs());
}
} // namespace Fortran::frontend
