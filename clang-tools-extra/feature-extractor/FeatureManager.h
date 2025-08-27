#pragma once

#include <tuple>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::tooling;
using namespace clang::ast_matchers;
using namespace llvm;

template <typename... Features> class FeatureManager {
  std::tuple<Features...> features;
  MatchFinder match_finder;

public:
  FeatureManager() {
    (
        [&]() {
          for (const auto &matcher : Features::Matchers)
            match_finder.addMatcher(matcher, &std::get<Features>(features));
        }(),
        ...);
  }

  MatchFinder *get_match_finder() { return &match_finder; }

  ~FeatureManager() {
    llvm::outs() << "\n";

    (
        [&]() {
          llvm::outs() << Features::get_title() << " : "
                       << std::get<Features>(features).get_result() << "\n";
        }(),
        ...);
  }
};
