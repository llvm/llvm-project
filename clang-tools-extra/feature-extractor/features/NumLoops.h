#pragma once

#include <array>
#include <cstddef>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "../utils.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
using namespace llvm;

class NumLoops : public MatchFinder::MatchCallback {
private:
  std::size_t num_loops{0};

public:
  static inline std::array Matchers = {forStmt().bind("forLoops"),
                                       whileStmt().bind("whileLoops")};

  virtual void run(const MatchFinder::MatchResult &result) override {
    auto context = result.Context;

    const auto fs = result.Nodes.getNodeAs<ForStmt>("forLoops");
    const auto ws = result.Nodes.getNodeAs<WhileStmt>("whileLoops");

    // We do not want to convert header files!
    if ((!fs ||
         !context->getSourceManager().isWrittenInMainFile(fs->getForLoc())) &&
        (!ws ||
         !context->getSourceManager().isWrittenInMainFile(ws->getWhileLoc())))
      return;

    num_loops++;
  }

  static const char *get_title() { return "num_loops"; }
  std::size_t get_result() const { return num_loops; }
};
