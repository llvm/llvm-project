//===--- ProTypeMemberInitCheck.h - clang-tidy-------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// \brief Implements C++ Core Guidelines Type.6.
///
/// Checks that every user-provided constructor value-initializes all class
/// members and base classes that would have undefined behavior otherwise. Also
/// check that any record types without user-provided default constructors are
/// value-initialized where used.
///
/// Members initialized through function calls in the body of the constructor
/// will result in false positives.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-pro-type-member-init.html
/// TODO: See if 'fixes' for false positives are optimized away by the compiler.
/// TODO: For classes with multiple constructors, make sure that we don't offer
///     multiple in-class initializer fixits for the same  member.
class ProTypeMemberInitCheck : public ClangTidyCheck {
public:
  ProTypeMemberInitCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  // Checks Type.6 part 1:
  // Issue a diagnostic for any constructor of a non-trivially-constructible
  // type that does not initialize all member variables.
  //
  // To fix: Write a data member initializer, or mention it in the member
  // initializer list.
  void checkMissingMemberInitializer(ASTContext &Context,
                                     const CXXRecordDecl &ClassDecl,
                                     const CXXConstructorDecl *Ctor);

  // A subtle side effect of Type.6 part 2:
  // Make sure to initialize trivially constructible base classes.
  void checkMissingBaseClassInitializer(const ASTContext &Context,
                                        const CXXRecordDecl &ClassDecl,
                                        const CXXConstructorDecl *Ctor);

  // Checks Type.6 part 2:
  // Issue a diagnostic when constructing an object of a trivially constructible
  // type without () or {} to initialize its members.
  //
  // To fix: Add () or {}.
  void checkUninitializedTrivialType(const ASTContext &Context,
                                     const VarDecl *Var);

  // Whether arrays need to be initialized or not. Default is false.
  bool IgnoreArrays;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_PRO_TYPE_MEMBER_INIT_H
