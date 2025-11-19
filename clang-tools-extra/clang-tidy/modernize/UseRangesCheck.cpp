//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRangesCheck.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <initializer_list>

// FixItHint - Let the docs script know that this class does provide fixits

namespace clang::tidy::modernize {

static constexpr const char *SingleRangeNames[] = {
    "all_of",
    "any_of",
    "none_of",
    "for_each",
    "find",
    "find_if",
    "find_if_not",
    "adjacent_find",
    "copy",
    "copy_if",
    "copy_backward",
    "move",
    "move_backward",
    "fill",
    "transform",
    "replace",
    "replace_if",
    "generate",
    "remove",
    "remove_if",
    "remove_copy",
    "remove_copy_if",
    "unique",
    "unique_copy",
    "sample",
    "partition_point",
    "lower_bound",
    "upper_bound",
    "equal_range",
    "binary_search",
    "push_heap",
    "pop_heap",
    "make_heap",
    "sort_heap",
    "next_permutation",
    "prev_permutation",
    "reverse",
    "reverse_copy",
    "shift_left",
    "shift_right",
    "is_partitioned",
    "partition",
    "partition_copy",
    "stable_partition",
    "sort",
    "stable_sort",
    "is_sorted",
    "is_sorted_until",
    "is_heap",
    "is_heap_until",
    "max_element",
    "min_element",
    "minmax_element",
    "uninitialized_copy",
    "uninitialized_fill",
    "uninitialized_move",
    "uninitialized_default_construct",
    "uninitialized_value_construct",
    "destroy",
};

static constexpr const char *TwoRangeNames[] = {
    "equal",
    "mismatch",
    "partial_sort_copy",
    "includes",
    "set_union",
    "set_intersection",
    "set_difference",
    "set_symmetric_difference",
    "merge",
    "lexicographical_compare",
    "find_end",
    "search",
    "is_permutation",
};

static constexpr const char *SinglePivotRangeNames[] = {"rotate", "rotate_copy",
                                                        "inplace_merge"};

namespace {
class StdReplacer : public utils::UseRangesCheck::Replacer {
public:
  explicit StdReplacer(SmallVector<UseRangesCheck::Signature> Signatures)
      : Signatures(std::move(Signatures)) {}
  std::optional<std::string>
  getReplaceName(const NamedDecl &OriginalName) const override {
    return ("std::ranges::" + OriginalName.getName()).str();
  }
  ArrayRef<UseRangesCheck::Signature>
  getReplacementSignatures() const override {
    return Signatures;
  }

private:
  SmallVector<UseRangesCheck::Signature> Signatures;
};

class StdAlgorithmReplacer : public StdReplacer {
  using StdReplacer::StdReplacer;
  std::optional<std::string>
  getHeaderInclusion(const NamedDecl & /*OriginalName*/) const override {
    return "<algorithm>";
  }
};

class StdNumericReplacer : public StdReplacer {
  using StdReplacer::StdReplacer;
  std::optional<std::string>
  getHeaderInclusion(const NamedDecl & /*OriginalName*/) const override {
    return "<numeric>";
  }
};
} // namespace

utils::UseRangesCheck::ReplacerMap UseRangesCheck::getReplacerMap() const {

  utils::UseRangesCheck::ReplacerMap Result;

  // template<typename Iter> Func(Iter first, Iter last,...).
  static const Signature SingleRangeArgs = {{0}};
  // template<typename Iter1, typename Iter2>
  // Func(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2,...).
  static const Signature TwoRangeArgs = {{0}, {2}};

  // template<typename Iter> Func(Iter first, Iter pivot, Iter last,...).
  static const Signature SinglePivotRange = {{0, 2}};

  static const Signature SingleRangeFunc[] = {SingleRangeArgs};

  static const Signature TwoRangeFunc[] = {TwoRangeArgs};

  static const Signature SinglePivotFunc[] = {SinglePivotRange};

  static const std::pair<ArrayRef<Signature>, ArrayRef<const char *>>
      AlgorithmNames[] = {{SingleRangeFunc, SingleRangeNames},
                          {TwoRangeFunc, TwoRangeNames},
                          {SinglePivotFunc, SinglePivotRangeNames}};
  SmallString<64> Buff;
  for (const auto &[Signatures, Values] : AlgorithmNames) {
    auto Replacer = llvm::makeIntrusiveRefCnt<StdAlgorithmReplacer>(
        SmallVector<UseRangesCheck::Signature>{Signatures});
    for (const auto &Name : Values) {
      Buff.assign({"::std::", Name});
      Result.try_emplace(Buff, Replacer);
    }
  }
  if (getLangOpts().CPlusPlus23)
    Result.try_emplace(
        "::std::iota",
        llvm::makeIntrusiveRefCnt<StdNumericReplacer>(
            SmallVector<UseRangesCheck::Signature>{std::begin(SingleRangeFunc),
                                                   std::end(SingleRangeFunc)}));
  return Result;
}

UseRangesCheck::UseRangesCheck(StringRef Name, ClangTidyContext *Context)
    : utils::UseRangesCheck(Name, Context),
      UseReversePipe(Options.get("UseReversePipe", false)) {}

void UseRangesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  utils::UseRangesCheck::storeOptions(Opts);
  Options.store(Opts, "UseReversePipe", UseReversePipe);
}

bool UseRangesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus20;
}
ArrayRef<std::pair<StringRef, StringRef>>
UseRangesCheck::getFreeBeginEndMethods() const {
  static const std::pair<StringRef, StringRef> Refs[] = {
      {"::std::begin", "::std::end"}, {"::std::cbegin", "::std::cend"}};
  return Refs;
}
std::optional<UseRangesCheck::ReverseIteratorDescriptor>
UseRangesCheck::getReverseDescriptor() const {
  static const std::pair<StringRef, StringRef> Refs[] = {
      {"::std::rbegin", "::std::rend"}, {"::std::crbegin", "::std::crend"}};
  return ReverseIteratorDescriptor{UseReversePipe ? "std::views::reverse"
                                                  : "std::ranges::reverse_view",
                                   "<ranges>", Refs, UseReversePipe};
}
} // namespace clang::tidy::modernize
