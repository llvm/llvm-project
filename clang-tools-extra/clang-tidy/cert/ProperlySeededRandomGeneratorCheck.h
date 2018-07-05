//===--- ProperlySeededRandomGeneratorCheck.h - clang-tidy-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_PROPERLY_SEEDED_RANDOM_GENERATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_PROPERLY_SEEDED_RANDOM_GENERATOR_H

#include "../ClangTidy.h"
#include <string>

namespace clang {
namespace tidy {
namespace cert {

/// Random number generator must be seeded properly.
///
/// A random number generator initialized with default value or a
/// constant expression is a security vulnerability.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cert-properly-seeded-random-generator.html
class ProperlySeededRandomGeneratorCheck : public ClangTidyCheck {
public:
  ProperlySeededRandomGeneratorCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  template <class T>
  void checkSeed(const ast_matchers::MatchFinder::MatchResult &Result,
                 const T *Func);

  std::string RawDisallowedSeedTypes;
  SmallVector<StringRef, 5> DisallowedSeedTypes;
};

} // namespace cert
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CERT_PROPERLY_SEEDED_RANDOM_GENERATOR_H
