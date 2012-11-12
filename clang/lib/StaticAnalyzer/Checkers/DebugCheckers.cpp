//==- DebugCheckers.cpp - Debugging Checkers ---------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a checkers that display debugging information.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/Support/Process.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// DominatorsTreeDumper
//===----------------------------------------------------------------------===//

namespace {
class DominatorsTreeDumper : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    if (AnalysisDeclContext *AC = mgr.getAnalysisDeclContext(D)) {
      DominatorTree dom;
      dom.buildDominatorTree(*AC);
      dom.dump();
    }
  }
};
}

void ento::registerDominatorsTreeDumper(CheckerManager &mgr) {
  mgr.registerChecker<DominatorsTreeDumper>();
}

//===----------------------------------------------------------------------===//
// LiveVariablesDumper
//===----------------------------------------------------------------------===//

namespace {
class LiveVariablesDumper : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    if (LiveVariables* L = mgr.getAnalysis<LiveVariables>(D)) {
      L->dumpBlockLiveness(mgr.getSourceManager());
    }
  }
};
}

void ento::registerLiveVariablesDumper(CheckerManager &mgr) {
  mgr.registerChecker<LiveVariablesDumper>();
}

//===----------------------------------------------------------------------===//
// CFGViewer
//===----------------------------------------------------------------------===//

namespace {
class CFGViewer : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    if (CFG *cfg = mgr.getCFG(D)) {
      cfg->viewCFG(mgr.getLangOpts());
    }
  }
};
}

void ento::registerCFGViewer(CheckerManager &mgr) {
  mgr.registerChecker<CFGViewer>();
}

//===----------------------------------------------------------------------===//
// CFGDumper
//===----------------------------------------------------------------------===//

namespace {
class CFGDumper : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager& mgr,
                        BugReporter &BR) const {
    if (CFG *cfg = mgr.getCFG(D)) {
      cfg->dump(mgr.getLangOpts(),
                llvm::sys::Process::StandardErrHasColors());
    }
  }
};
}

void ento::registerCFGDumper(CheckerManager &mgr) {
  mgr.registerChecker<CFGDumper>();
}

//===----------------------------------------------------------------------===//
// CallGraphViewer
//===----------------------------------------------------------------------===//

namespace {
class CallGraphViewer : public Checker< check::ASTDecl<TranslationUnitDecl> > {
public:
  void checkASTDecl(const TranslationUnitDecl *TU, AnalysisManager& mgr,
                    BugReporter &BR) const {
    CallGraph CG;
    CG.addToCallGraph(const_cast<TranslationUnitDecl*>(TU));
    CG.viewGraph();
  }
};
}

void ento::registerCallGraphViewer(CheckerManager &mgr) {
  mgr.registerChecker<CallGraphViewer>();
}

//===----------------------------------------------------------------------===//
// CallGraphDumper
//===----------------------------------------------------------------------===//

namespace {
class CallGraphDumper : public Checker< check::ASTDecl<TranslationUnitDecl> > {
public:
  void checkASTDecl(const TranslationUnitDecl *TU, AnalysisManager& mgr,
                    BugReporter &BR) const {
    CallGraph CG;
    CG.addToCallGraph(const_cast<TranslationUnitDecl*>(TU));
    CG.dump();
  }
};
}

void ento::registerCallGraphDumper(CheckerManager &mgr) {
  mgr.registerChecker<CallGraphDumper>();
}


//===----------------------------------------------------------------------===//
// ConfigDumper
//===----------------------------------------------------------------------===//

namespace {
class ConfigDumper : public Checker< check::EndOfTranslationUnit > {
public:
  void checkEndOfTranslationUnit(const TranslationUnitDecl *TU,
                                 AnalysisManager& mgr,
                                 BugReporter &BR) const {

    const AnalyzerOptions::ConfigTable &Config = mgr.options.Config;
    AnalyzerOptions::ConfigTable::const_iterator I =
      Config.begin(), E = Config.end();

    std::vector<StringRef> Keys;
    for (; I != E ; ++I) { Keys.push_back(I->getKey()); }
    sort(Keys.begin(), Keys.end());
    
    llvm::errs() << "[config]\n";
    for (unsigned i = 0, n = Keys.size(); i < n ; ++i) {
      StringRef Key = Keys[i];
      I = Config.find(Key);
      llvm::errs() << Key << " = " << I->second << '\n';
    }
    llvm::errs() << "[stats]\n" << "num-entries = " << Keys.size() << '\n';
  }
};
}

void ento::registerConfigDumper(CheckerManager &mgr) {
  mgr.registerChecker<ConfigDumper>();
}
