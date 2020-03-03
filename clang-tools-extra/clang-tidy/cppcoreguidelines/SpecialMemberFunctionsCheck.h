//===--- SpecialMemberFunctionsCheck.h - clang-tidy--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SPECIAL_MEMBER_FUNCTIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SPECIAL_MEMBER_FUNCTIONS_H

#include "../ClangTidy.h"

#include "llvm/ADT/DenseMapInfo.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// Checks for classes where some, but not all, of the special member functions
/// are defined.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines-special-member-functions.html
class SpecialMemberFunctionsCheck : public ClangTidyCheck {
public:
  SpecialMemberFunctionsCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

  enum class SpecialMemberFunctionKind : uint8_t {
    Destructor,
    DefaultDestructor,
    NonDefaultDestructor,
    CopyConstructor,
    CopyAssignment,
    MoveConstructor,
    MoveAssignment
  };

  using ClassDefId = std::pair<SourceLocation, std::string>;

  using ClassDefiningSpecialMembersMap =
      llvm::DenseMap<ClassDefId,
                     llvm::SmallVector<SpecialMemberFunctionKind, 5>>;

private:
  void checkForMissingMembers(
      const ClassDefId &ID,
      llvm::ArrayRef<SpecialMemberFunctionKind> DefinedSpecialMembers);

  const bool AllowMissingMoveFunctions;
  const bool AllowSoleDefaultDtor;
  ClassDefiningSpecialMembersMap ClassWithSpecialMembers;
};

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang

namespace llvm {
/// Specialisation of DenseMapInfo to allow ClassDefId objects in DenseMaps
/// FIXME: Move this to the corresponding cpp file as is done for
/// clang-tidy/readability/IdentifierNamingCheck.cpp.
template <>
struct DenseMapInfo<
    clang::tidy::cppcoreguidelines::SpecialMemberFunctionsCheck::ClassDefId> {
  using ClassDefId =
      clang::tidy::cppcoreguidelines::SpecialMemberFunctionsCheck::ClassDefId;

  static inline ClassDefId getEmptyKey() {
    return ClassDefId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-1)),
        "EMPTY");
  }

  static inline ClassDefId getTombstoneKey() {
    return ClassDefId(
        clang::SourceLocation::getFromRawEncoding(static_cast<unsigned>(-2)),
        "TOMBSTONE");
  }

  static unsigned getHashValue(ClassDefId Val) {
    assert(Val != getEmptyKey() && "Cannot hash the empty key!");
    assert(Val != getTombstoneKey() && "Cannot hash the tombstone key!");

    std::hash<ClassDefId::second_type> SecondHash;
    return Val.first.getRawEncoding() + SecondHash(Val.second);
  }

  static bool isEqual(const ClassDefId &LHS, const ClassDefId &RHS) {
    if (RHS == getEmptyKey())
      return LHS == getEmptyKey();
    if (RHS == getTombstoneKey())
      return LHS == getTombstoneKey();
    return LHS == RHS;
  }
};

} // namespace llvm

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_CPPCOREGUIDELINES_SPECIAL_MEMBER_FUNCTIONS_H
