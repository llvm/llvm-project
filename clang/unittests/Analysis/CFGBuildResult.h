//===- unittests/Analysis/CFGBuildResult.h - CFG tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CFG.h"
#include "clang/Tooling/Tooling.h"
#include <memory>

namespace clang {
namespace analysis {

class BuildResult {
public:
  enum Status {
    ToolFailed,
    ToolRan,
    SawFunctionBody,
    BuiltCFG,
  };

  BuildResult(Status S, std::unique_ptr<CFG> Cfg = nullptr,
              std::unique_ptr<ASTUnit> AST = nullptr)
      : S(S), Cfg(std::move(Cfg)), AST(std::move(AST)) {}

  Status getStatus() const { return S; }
  CFG *getCFG() const { return Cfg.get(); }
  ASTUnit *getAST() const { return AST.get(); }

private:
  Status S;
  std::unique_ptr<CFG> Cfg;
  std::unique_ptr<ASTUnit> AST;
};

class CFGCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  CFGCallback(std::unique_ptr<ASTUnit> AST) : AST(std::move(AST)) {}

  std::unique_ptr<ASTUnit> AST;
  BuildResult TheBuildResult = BuildResult::ToolRan;
  CFG::BuildOptions Options;

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    Stmt *Body = Func->getBody();
    if (!Body)
      return;
    TheBuildResult = BuildResult::SawFunctionBody;
    Options.AddImplicitDtors = true;
    if (std::unique_ptr<CFG> Cfg =
            CFG::buildCFG(nullptr, Body, Result.Context, Options))
      TheBuildResult = {BuildResult::BuiltCFG, std::move(Cfg), std::move(AST)};
  }
};

inline BuildResult BuildCFG(const char *Code, CFG::BuildOptions Options = {}) {
  std::vector<std::string> Args = {"-std=c++11",
                                   "-fno-delayed-template-parsing"};
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(Code, Args);
  if (!AST)
    return BuildResult::ToolFailed;

  CFGCallback Callback(std::move(AST));
  Callback.Options = Options;
  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(ast_matchers::functionDecl().bind("func"), &Callback);

  Finder.matchAST(Callback.AST->getASTContext());
  return std::move(Callback.TheBuildResult);
}

} // namespace analysis
} // namespace clang
