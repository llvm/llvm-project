//===- unittests/Analysis/CFGBuildResult.h - CFG tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/CFG.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"

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

  BuildResult(Status S, std::unique_ptr<CFG> Cfg = nullptr)
      : S(S), Cfg(std::move(Cfg)) {}

  Status getStatus() const { return S; }
  CFG *getCFG() const { return Cfg.get(); }

private:
  Status S;
  std::unique_ptr<CFG> Cfg;
};

class CFGCallback : public ast_matchers::MatchFinder::MatchCallback {
public:
  BuildResult TheBuildResult = BuildResult::ToolRan;

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override {
    const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
    Stmt *Body = Func->getBody();
    if (!Body)
      return;
    TheBuildResult = BuildResult::SawFunctionBody;
    CFG::BuildOptions Options;
    Options.AddImplicitDtors = true;
    if (std::unique_ptr<CFG> Cfg =
            CFG::buildCFG(nullptr, Body, Result.Context, Options))
      TheBuildResult = {BuildResult::BuiltCFG, std::move(Cfg)};
  }
};

inline BuildResult BuildCFG(const char *Code) {
  CFGCallback Callback;

  ast_matchers::MatchFinder Finder;
  Finder.addMatcher(ast_matchers::functionDecl().bind("func"), &Callback);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  std::vector<std::string> Args = {"-std=c++11",
                                   "-fno-delayed-template-parsing"};
  if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args))
    return BuildResult::ToolFailed;
  return std::move(Callback.TheBuildResult);
}

} // namespace analysis
} // namespace clang
