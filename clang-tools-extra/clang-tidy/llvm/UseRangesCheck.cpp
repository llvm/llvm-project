//===--- UseRangesCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRangesCheck.h"

namespace clang::tidy::llvm_check {

namespace {

class StdToLLVMReplacer : public utils::UseRangesCheck::Replacer {
public:
  explicit StdToLLVMReplacer(
      ArrayRef<utils::UseRangesCheck::Signature> Signatures)
      : Signatures(Signatures) {}

  ArrayRef<utils::UseRangesCheck::Signature>
  getReplacementSignatures() const override {
    return Signatures;
  }

  std::optional<std::string>
  getReplaceName(const NamedDecl &OriginalName) const override {
    return ("llvm::" + OriginalName.getName()).str();
  }

  std::optional<std::string>
  getHeaderInclusion(const NamedDecl &) const override {
    return "llvm/ADT/STLExtras.h";
  }

private:
  ArrayRef<utils::UseRangesCheck::Signature> Signatures;
};

} // namespace

utils::UseRangesCheck::ReplacerMap UseRangesCheck::getReplacerMap() const {
  ReplacerMap Results;

  static const Signature SingleSig = {{0}};
  static const Signature TwoSig = {{0}, {2}};

  const auto AddStdToLLVM =
      [&Results](llvm::IntrusiveRefCntPtr<Replacer> Replacer,
                 std::initializer_list<StringRef> Names) {
        for (const auto &Name : Names) {
          Results.try_emplace(("::std::" + Name).str(), Replacer);
        }
      };

  // Single range algorithms
  AddStdToLLVM(llvm::makeIntrusiveRefCnt<StdToLLVMReplacer>(SingleSig),
               {"all_of",      "any_of",
                "none_of",     "for_each",
                "find",        "find_if",
                "find_if_not", "fill",
                "count",       "count_if",
                "copy",        "copy_if",
                "transform",   "replace",
                "remove_if",   "stable_sort",
                "partition",   "partition_point",
                "is_sorted",   "min_element",
                "max_element", "binary_search",
                "lower_bound", "upper_bound",
                "unique",      "uninitialized_copy"});

  // Two range algorithms
  AddStdToLLVM(llvm::makeIntrusiveRefCnt<StdToLLVMReplacer>(TwoSig),
               {"equal", "mismatch", "includes"});

  return Results;
}

UseRangesCheck::UseRangesCheck(StringRef Name, ClangTidyContext *Context)
    : utils::UseRangesCheck(Name, Context) {}

DiagnosticBuilder UseRangesCheck::createDiag(const CallExpr &Call) {
  return diag(Call.getBeginLoc(), "use an LLVM range-based algorithm");
}

ArrayRef<std::pair<StringRef, StringRef>>
UseRangesCheck::getFreeBeginEndMethods() const {
  static constexpr std::pair<StringRef, StringRef> Refs[] = {
      {"::std::begin", "::std::end"},
      {"::std::cbegin", "::std::cend"},
      {"::std::rbegin", "::std::rend"},
      {"::std::crbegin", "::std::crend"},
  };
  return Refs;
}

} // namespace clang::tidy::llvm_check
