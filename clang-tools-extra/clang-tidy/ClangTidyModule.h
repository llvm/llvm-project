//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULE_H

#include "ClangTidyOptions.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>

namespace clang::tidy {

class ClangTidyCheck;
class ClangTidyContext;

/// A collection of \c ClangTidyCheckFactory instances.
///
/// All clang-tidy modules register their check factories with an instance of
/// this object.
class ClangTidyCheckFactories {
public:
  using CheckFactory = std::function<std::unique_ptr<ClangTidyCheck>(
      llvm::StringRef Name, ClangTidyContext *Context)>;

  /// Registers check \p Factory with name \p Name.
  ///
  /// For all checks that have default constructors, use \c registerCheck.
  void registerCheckFactory(llvm::StringRef Name, CheckFactory Factory);

  /// Registers the \c CheckType with the name \p Name.
  ///
  /// This method should be used for all \c ClangTidyChecks that don't require
  /// constructor parameters.
  ///
  /// For example, if have a clang-tidy check like:
  /// \code
  /// class MyTidyCheck : public ClangTidyCheck {
  ///   void registerMatchers(ast_matchers::MatchFinder *Finder) override {
  ///     ..
  ///   }
  /// };
  /// \endcode
  /// you can register it with:
  /// \code
  /// class MyModule : public ClangTidyModule {
  ///   void addCheckFactories(ClangTidyCheckFactories &Factories) override {
  ///     Factories.registerCheck<MyTidyCheck>("myproject-my-check");
  ///   }
  /// };
  /// \endcode
  template <typename CheckType> void registerCheck(llvm::StringRef CheckName) {
    registerCheckFactory(CheckName,
                         [](llvm::StringRef Name, ClangTidyContext *Context) {
                           return std::make_unique<CheckType>(Name, Context);
                         });
  }

  /// Create instances of checks that are enabled.
  std::vector<std::unique_ptr<ClangTidyCheck>>
  createChecks(ClangTidyContext *Context) const;

  /// Create instances of checks that are enabled for the current Language.
  std::vector<std::unique_ptr<ClangTidyCheck>>
  createChecksForLanguage(ClangTidyContext *Context) const;

  using FactoryMap = llvm::StringMap<CheckFactory>;
  FactoryMap::const_iterator begin() const { return Factories.begin(); }
  FactoryMap::const_iterator end() const { return Factories.end(); }
  bool empty() const { return Factories.empty(); }

private:
  FactoryMap Factories;
};

/// A clang-tidy module groups a number of \c ClangTidyChecks and gives
/// them a prefixed name.
class ClangTidyModule {
public:
  virtual ~ClangTidyModule() {}

  /// Implement this function in order to register all \c CheckFactories
  /// belonging to this module.
  virtual void addCheckFactories(ClangTidyCheckFactories &CheckFactories) = 0;

  /// Gets default options for checks defined in this module.
  virtual ClangTidyOptions getModuleOptions();
};

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CLANGTIDYMODULE_H
