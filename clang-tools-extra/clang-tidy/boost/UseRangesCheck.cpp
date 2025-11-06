//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRangesCheck.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <initializer_list>
#include <optional>
#include <string>

// FixItHint - Let the docs script know that this class does provide fixits

namespace clang::tidy::boost {

namespace {
/// Base replacer that handles the boost include path and namespace
class BoostReplacer : public UseRangesCheck::Replacer {
public:
  BoostReplacer(ArrayRef<UseRangesCheck::Signature> Signatures,
                bool IncludeSystem)
      : Signatures(Signatures), IncludeSystem(IncludeSystem) {}

  ArrayRef<UseRangesCheck::Signature> getReplacementSignatures() const final {
    return Signatures;
  }

  virtual std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const = 0;

  virtual std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const = 0;

  std::optional<std::string>
  getReplaceName(const NamedDecl &OriginalName) const final {
    auto [Namespace, Function] = getBoostName(OriginalName);
    return ("boost::" + Namespace + (Namespace.empty() ? "" : "::") + Function)
        .str();
  }

  std::optional<std::string>
  getHeaderInclusion(const NamedDecl &OriginalName) const final {
    auto [Path, HeaderName] = getBoostHeader(OriginalName);
    return ((IncludeSystem ? "<boost/" : "boost/") + Path +
            (Path.empty() ? "" : "/") + HeaderName +
            (IncludeSystem ? ".hpp>" : ".hpp"))
        .str();
  }

private:
  SmallVector<UseRangesCheck::Signature> Signatures;
  bool IncludeSystem;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<FUNC_NAME>.hpp` and the function is named
/// `boost::range::<FUNC_NAME>`
class BoostRangeAlgorithmReplacer : public BoostReplacer {
public:
  using BoostReplacer::BoostReplacer;

  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"range", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const override {
    return {"range/algorithm", OriginalName.getName()};
  }
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<CUSTOM_HEADER>.hpp` and the function is named
/// `boost::range::<FUNC_NAME>`
class CustomBoostAlgorithmHeaderReplacer : public BoostRangeAlgorithmReplacer {
public:
  CustomBoostAlgorithmHeaderReplacer(
      StringRef HeaderName, ArrayRef<UseRangesCheck::Signature> Signatures,
      bool IncludeSystem)
      : BoostRangeAlgorithmReplacer(Signatures, IncludeSystem),
        HeaderName(HeaderName) {}

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl & /*OriginalName*/) const override {
    return {"range/algorithm", HeaderName};
  }

private:
  StringRef HeaderName;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<SUB_HEADER>.hpp` and the function is named
/// `boost::algorithm::<FUNC_NAME>`
class BoostAlgorithmReplacer : public BoostReplacer {
public:
  BoostAlgorithmReplacer(StringRef SubHeader,
                         ArrayRef<UseRangesCheck::Signature> Signatures,
                         bool IncludeSystem)
      : BoostReplacer(Signatures, IncludeSystem),
        SubHeader(("algorithm/" + SubHeader).str()) {}
  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"algorithm", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const override {
    return {SubHeader, OriginalName.getName()};
  }

private:
  std::string SubHeader;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<SUB_HEADER>/<HEADER_NAME>.hpp` and the function is named
/// `boost::algorithm::<FUNC_NAME>`
class CustomBoostAlgorithmReplacer : public BoostReplacer {
public:
  CustomBoostAlgorithmReplacer(StringRef SubHeader, StringRef HeaderName,
                               ArrayRef<UseRangesCheck::Signature> Signatures,
                               bool IncludeSystem)
      : BoostReplacer(Signatures, IncludeSystem),
        SubHeader(("algorithm/" + SubHeader).str()), HeaderName(HeaderName) {}
  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"algorithm", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl & /*OriginalName*/) const override {
    return {SubHeader, HeaderName};
  }

private:
  std::string SubHeader;
  StringRef HeaderName;
};

/// A Replacer that is used for functions that just call a new overload
class MakeOverloadReplacer : public UseRangesCheck::Replacer {
public:
  explicit MakeOverloadReplacer(ArrayRef<UseRangesCheck::Signature> Signatures)
      : Signatures(Signatures) {}

  ArrayRef<UseRangesCheck::Signature>
  getReplacementSignatures() const override {
    return Signatures;
  }

  std::optional<std::string>
  getReplaceName(const NamedDecl & /* OriginalName */) const override {
    return std::nullopt;
  }

  std::optional<std::string>
  getHeaderInclusion(const NamedDecl & /* OriginalName */) const override {
    return std::nullopt;
  }

private:
  SmallVector<UseRangesCheck::Signature> Signatures;
};

/// A replacer that replaces functions with an equivalent named function in the
/// root boost namespace
class FixedBoostReplace : public BoostReplacer {
public:
  FixedBoostReplace(StringRef Header,
                    ArrayRef<UseRangesCheck::Signature> Signatures,
                    bool IncludeBoostSystem)
      : BoostReplacer(Signatures, IncludeBoostSystem), Header(Header) {}

  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {{}, OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl & /* OriginalName */) const override {
    return {{}, Header};
  }

private:
  StringRef Header;
};

} // namespace

utils::UseRangesCheck::ReplacerMap UseRangesCheck::getReplacerMap() const {

  ReplacerMap Results;
  static const Signature SingleSig = {{0}};
  static const Signature TwoSig = {{0}, {2}};
  const auto AddFrom =
      [&Results](llvm::IntrusiveRefCntPtr<UseRangesCheck::Replacer> Replacer,
                 std::initializer_list<StringRef> Names, StringRef Prefix) {
        llvm::SmallString<64> Buffer;
        for (const auto &Name : Names) {
          Buffer.assign({"::", Prefix, (Prefix.empty() ? "" : "::"), Name});
          Results.try_emplace(Buffer, Replacer);
        }
      };

  const auto AddFromStd =
      [&](llvm::IntrusiveRefCntPtr<UseRangesCheck::Replacer> Replacer,
          std::initializer_list<StringRef> Names) {
        AddFrom(Replacer, Names, "std");
      };

  const auto AddFromBoost =
      [&](llvm::IntrusiveRefCntPtr<UseRangesCheck::Replacer> Replacer,
          std::initializer_list<
              std::pair<StringRef, std::initializer_list<StringRef>>>
              NamespaceAndNames) {
        for (auto [Namespace, Names] : NamespaceAndNames)
          AddFrom(Replacer, Names,
                  SmallString<64>{"boost", (Namespace.empty() ? "" : "::"),
                                  Namespace});
      };

  AddFromStd(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
                 "set_algorithm", TwoSig, IncludeBoostSystem),
             {"includes", "set_union", "set_intersection", "set_difference",
              "set_symmetric_difference"});

  AddFromStd(llvm::makeIntrusiveRefCnt<BoostRangeAlgorithmReplacer>(
                 SingleSig, IncludeBoostSystem),
             {"unique",         "lower_bound",   "stable_sort",
              "equal_range",    "remove_if",     "sort",
              "random_shuffle", "remove_copy",   "stable_partition",
              "remove_copy_if", "count",         "copy_backward",
              "reverse_copy",   "adjacent_find", "remove",
              "upper_bound",    "binary_search", "replace_copy_if",
              "for_each",       "generate",      "count_if",
              "min_element",    "reverse",       "replace_copy",
              "fill",           "unique_copy",   "transform",
              "copy",           "replace",       "find",
              "replace_if",     "find_if",       "partition",
              "max_element"});

  AddFromStd(llvm::makeIntrusiveRefCnt<BoostRangeAlgorithmReplacer>(
                 TwoSig, IncludeBoostSystem),
             {"find_end", "merge", "partial_sort_copy", "find_first_of",
              "search", "lexicographical_compare", "equal", "mismatch"});

  AddFromStd(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
                 "permutation", SingleSig, IncludeBoostSystem),
             {"next_permutation", "prev_permutation"});

  AddFromStd(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
                 "heap_algorithm", SingleSig, IncludeBoostSystem),
             {"push_heap", "pop_heap", "make_heap", "sort_heap"});

  AddFromStd(llvm::makeIntrusiveRefCnt<BoostAlgorithmReplacer>(
                 "cxx11", SingleSig, IncludeBoostSystem),
             {"copy_if", "is_permutation", "is_partitioned", "find_if_not",
              "partition_copy", "any_of", "iota", "all_of", "partition_point",
              "is_sorted", "none_of"});

  AddFromStd(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmReplacer>(
                 "cxx11", "is_sorted", SingleSig, IncludeBoostSystem),
             {"is_sorted_until"});

  AddFromStd(llvm::makeIntrusiveRefCnt<FixedBoostReplace>(
                 "range/numeric", SingleSig, IncludeBoostSystem),
             {"accumulate", "partial_sum", "adjacent_difference"});

  if (getLangOpts().CPlusPlus17)
    AddFromStd(llvm::makeIntrusiveRefCnt<BoostAlgorithmReplacer>(
                   "cxx17", SingleSig, IncludeBoostSystem),
               {"reduce"});

  AddFromBoost(llvm::makeIntrusiveRefCnt<MakeOverloadReplacer>(SingleSig),
               {{"algorithm",
                 {"reduce",
                  "find_backward",
                  "find_not_backward",
                  "find_if_backward",
                  "find_if_not_backward",
                  "hex",
                  "hex_lower",
                  "unhex",
                  "is_partitioned_until",
                  "is_palindrome",
                  "copy_if",
                  "copy_while",
                  "copy_until",
                  "copy_if_while",
                  "copy_if_until",
                  "is_permutation",
                  "is_partitioned",
                  "one_of",
                  "one_of_equal",
                  "find_if_not",
                  "partition_copy",
                  "any_of",
                  "any_of_equal",
                  "iota",
                  "all_of",
                  "all_of_equal",
                  "partition_point",
                  "is_sorted_until",
                  "is_sorted",
                  "is_increasing",
                  "is_decreasing",
                  "is_strictly_increasing",
                  "is_strictly_decreasing",
                  "none_of",
                  "none_of_equal",
                  "clamp_range"}}});

  AddFromBoost(
      llvm::makeIntrusiveRefCnt<MakeOverloadReplacer>(TwoSig),
      {{"algorithm", {"apply_permutation", "apply_reverse_permutation"}}});

  return Results;
}

UseRangesCheck::UseRangesCheck(StringRef Name, ClangTidyContext *Context)
    : utils::UseRangesCheck(Name, Context),
      IncludeBoostSystem(Options.get("IncludeBoostSystem", true)),
      UseReversePipe(Options.get("UseReversePipe", false)) {}

void UseRangesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  utils::UseRangesCheck::storeOptions(Opts);
  Options.store(Opts, "IncludeBoostSystem", IncludeBoostSystem);
  Options.store(Opts, "UseReversePipe", UseReversePipe);
}

DiagnosticBuilder UseRangesCheck::createDiag(const CallExpr &Call) {
  DiagnosticBuilder D =
      diag(Call.getBeginLoc(), "use a %0 version of this algorithm");
  D << (Call.getDirectCallee()->isInStdNamespace() ? "boost" : "ranged");
  return D;
}
ArrayRef<std::pair<StringRef, StringRef>>
UseRangesCheck::getFreeBeginEndMethods() const {
  static const std::pair<StringRef, StringRef> Refs[] = {
      {"::std::begin", "::std::end"},
      {"::std::cbegin", "::std::cend"},
      {"::boost::range_adl_barrier::begin", "::boost::range_adl_barrier::end"},
      {"::boost::range_adl_barrier::const_begin",
       "::boost::range_adl_barrier::const_end"},
  };
  return Refs;
}
std::optional<UseRangesCheck::ReverseIteratorDescriptor>
UseRangesCheck::getReverseDescriptor() const {
  static const std::pair<StringRef, StringRef> Refs[] = {
      {"::std::rbegin", "::std::rend"},
      {"::std::crbegin", "::std::crend"},
      {"::boost::rbegin", "::boost::rend"},
      {"::boost::const_rbegin", "::boost::const_rend"},
  };
  return ReverseIteratorDescriptor{
      UseReversePipe ? "boost::adaptors::reversed" : "boost::adaptors::reverse",
      IncludeBoostSystem ? "<boost/range/adaptor/reversed.hpp>"
                         : "boost/range/adaptor/reversed.hpp",
      Refs, UseReversePipe};
}
} // namespace clang::tidy::boost
