//===---- Query.cpp - clang-query query -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QuerySession.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;

namespace clang {
namespace query {

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << ErrStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER          "
        "Match the loaded ASTs against the given matcher.\n"
        "  let NAME MATCHER, l NAME MATCHER  "
        "Give a matcher expression a name, to be used later\n"
        "                                    "
        "as part of other expressions.\n"
        "  set bind-root (true|false)        "
        "Set whether to bind the root matcher to \"root\".\n"
        "  set output (diag|print|dump)      "
        "Set whether to print bindings as diagnostics,\n"
        "                                    "
        "AST pretty prints or AST dumps.\n"
        "  quit                              "
        "Terminates the query session.\n\n";
  return true;
}

bool QuitQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  QS.Terminate = true;
  return true;
}

namespace {

struct CollectBoundNodes : MatchFinder::MatchCallback {
  std::vector<BoundNodes> &Bindings;
  CollectBoundNodes(std::vector<BoundNodes> &Bindings) : Bindings(Bindings) {}
  void run(const MatchFinder::MatchResult &Result) override {
    Bindings.push_back(Result.Nodes);
  }
};

} // namespace

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  unsigned MatchCount = 0;

  for (auto &AST : QS.ASTs) {
    MatchFinder Finder;
    std::vector<BoundNodes> Matches;
    DynTypedMatcher MaybeBoundMatcher = Matcher;
    if (QS.BindRoot) {
      llvm::Optional<DynTypedMatcher> M = Matcher.tryBind("root");
      if (M)
        MaybeBoundMatcher = *M;
    }
    CollectBoundNodes Collect(Matches);
    if (!Finder.addDynamicMatcher(MaybeBoundMatcher, &Collect)) {
      OS << "Not a valid top-level matcher.\n";
      return false;
    }
    Finder.matchAST(AST->getASTContext());

    for (auto MI = Matches.begin(), ME = Matches.end(); MI != ME; ++MI) {
      OS << "\nMatch #" << ++MatchCount << ":\n\n";

      for (auto BI = MI->getMap().begin(), BE = MI->getMap().end(); BI != BE;
           ++BI) {
        switch (QS.OutKind) {
        case OK_Diag: {
          clang::SourceRange R = BI->second.getSourceRange();
          if (R.isValid()) {
            TextDiagnostic TD(OS, AST->getASTContext().getLangOpts(),
                              &AST->getDiagnostics().getDiagnosticOptions());
            TD.emitDiagnostic(R.getBegin(), DiagnosticsEngine::Note,
                              "\"" + BI->first + "\" binds here",
                              CharSourceRange::getTokenRange(R), None,
                              &AST->getSourceManager());
          }
          break;
        }
        case OK_Print: {
          OS << "Binding for \"" << BI->first << "\":\n";
          BI->second.print(OS, AST->getASTContext().getPrintingPolicy());
          OS << "\n";
          break;
        }
        case OK_Dump: {
          OS << "Binding for \"" << BI->first << "\":\n";
          BI->second.dump(OS, AST->getSourceManager());
          OS << "\n";
          break;
        }
        }
      }

      if (MI->getMap().empty())
        OS << "No bindings.\n";
    }
  }

  OS << MatchCount << (MatchCount == 1 ? " match.\n" : " matches.\n");
  return true;
}

bool LetQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  if (Value) {
    QS.NamedValues[Name] = Value;
  } else {
    QS.NamedValues.erase(Name);
  }
  return true;
}

#ifndef _MSC_VER
const QueryKind SetQueryKind<bool>::value;
const QueryKind SetQueryKind<OutputKind>::value;
#endif

} // namespace query
} // namespace clang
