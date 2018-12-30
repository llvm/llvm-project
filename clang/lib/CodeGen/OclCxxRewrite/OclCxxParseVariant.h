//===- OclCxxParseVariant.h - Parse variant in OCLC++ demangler -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
//
//===----------------------------------------------------------------------===//


#ifndef CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPARSEVARIANT_H
#define CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPARSEVARIANT_H

#include <cassert>
#include <type_traits>
#include <utility>


namespace oclcxx {
namespace adaptation {

// -----------------------------------------------------------------------------
// HELPERS FOR ALIGNED UNION STORAGE
// -----------------------------------------------------------------------------
// Replacement for std::aligned_union which is curiously not defined in
// libstdc++ until version 4 (gcc 5).
// TODO: Use std::aligned_union once all used compilers will support it.

/// \brief Helper trait calculating maximal value from given compile-time
///        constants.
template <std::size_t Val, std::size_t ... Vals>
struct MaxHelper
  : std::integral_constant<std::size_t, (Val > MaxHelper<Vals ...>::value
                                          ? Val
                                          : MaxHelper<Vals ...>::value)> {};

template <size_t ValA, size_t ValB>
struct MaxHelper<ValA, ValB>
  : std::integral_constant<std::size_t, (ValA > ValB
                                          ? ValA
                                          : ValB)> {};


/// \brief Trait generating a POD type that has size and suitable alignment to
///        store object of any type listed in TypesT.
template <size_t Len, typename ... TypesT>
struct AlignedUnion
{
  static const std::size_t alignment_value =
    MaxHelper<std::alignment_of<TypesT>::value ...>::value;
  using type = typename std::aligned_storage<
                  MaxHelper<Len, sizeof(TypesT) ...>::value,
                  alignment_value>::type;
};

// -----------------------------------------------------------------------------
// PARSE VARIANT HELPERS
// -----------------------------------------------------------------------------

/// \brief Scorer for input value (in relation to stored type).
template <typename ValT, typename STypeT,
          typename RawValT = typename std::decay<ValT>::type,
          typename RawSTypeT = typename std::decay<STypeT>::type,
          bool Cond = std::is_constructible<RawSTypeT, ValT &&>::value,
          int CondId = 0>
struct ParseVarValScorer;

// [T] Condition 0: Can we construct storage type from value type.
// Check whether input type is the same as storage type (max score) or
// whether input type is converted to class/struct/union type.
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, true, 0> {
  static const int Value = std::is_same<RawValT, RawSTypeT>::value
    ? 6
    : (std::is_class<RawSTypeT>::value || std::is_union<RawSTypeT>::value
      ? 5
      : ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT,
                          std::is_enum<RawValT>::value, 1>::Value);
};

// [F] Condition 0: Can we construct storage type from value type.
// Score is 0, if we cannot (no match).
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, false, 0> {
  static const int Value = 0;
};

// [T] Condition 1: Input value is enum.
// Switch to underlying type and do the arithmetic tests.
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, true, 1> {
  static const int Value =
    ParseVarValScorer<typename std::underlying_type<RawValT>::type, STypeT,
                      typename std::underlying_type<RawValT>::type, RawSTypeT,
                      false, 1>::Value;
};

// [F] Condition 1: Input value is enum.
// Check if we have simple integer-integer or fp-fp conversions.
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, false, 1> {
  static const int Value =
    ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT,
                      (std::is_integral<RawValT>::value &&
                       std::is_integral<RawSTypeT>::value) ||
                      (std::is_floating_point<RawValT>::value &&
                       std::is_floating_point<RawSTypeT>::value), 2>::Value;
};

// [T] Condition 2: Simple arithmetic conversions.
// Switch to underlying type and do the arithmetic tests.
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, true, 2> {
  static const int Value = sizeof(RawValT) == sizeof(RawSTypeT)
    ? (std::is_signed<RawValT>::value == std::is_signed<RawSTypeT>::value
      ? 4
      : 3)
    : (sizeof(RawValT) < sizeof(RawSTypeT)
      ? (std::is_unsigned<RawValT>::value
        ? 4
        : 3)
      : 2);
};

// [F] Condition 2: Simple arithmetic conversions.
template <typename ValT, typename STypeT, typename RawValT, typename RawSTypeT>
struct ParseVarValScorer<ValT, STypeT, RawValT, RawSTypeT, false, 2> {
  static const int Value = 1;
};


/// \brief Input value matcher for ParseVariant.
///
/// This is a raw helper, please use ParseVarValMatcherT alias instead.
template <typename ValueT, int Score, int Idx, int BestIdx,
          typename BestMatchTypeT, typename ... StoredTypesT>
struct ParseVarValMatcher;

// Value got maximal score (early return).
template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {
  static const bool Value = true;
  using Type = typename std::decay<BestMatchTypeT>::type;
  static const int Index = BestIdx;
};

template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT,
          typename StoredTypeT, typename ... StoredTypesT>
struct ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT,
                          StoredTypeT, StoredTypesT ...>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

// Any non-zero score is good at the end (return first with maximal score).
template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 5, Idx, BestIdx, BestMatchTypeT>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 4, Idx, BestIdx, BestMatchTypeT>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 3, Idx, BestIdx, BestMatchTypeT>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 2, Idx, BestIdx, BestMatchTypeT>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 1, Idx, BestIdx, BestMatchTypeT>
  : ParseVarValMatcher<ValueT, 6, Idx, BestIdx, BestMatchTypeT> {};

// Zero score at the end (matched failed).
template <typename ValueT, int Idx, int BestIdx, typename BestMatchTypeT>
struct ParseVarValMatcher<ValueT, 0, Idx, BestIdx, BestMatchTypeT> {
  static const bool Value = false;
  static const int Index = -1;
};

// Scan recursively through stored types.
template <typename ValueT, int Score, int Idx, int BestIdx,
          typename BestMatchTypeT, typename StoredTypeT,
          typename ... StoredTypesT>
struct ParseVarValMatcher<ValueT, Score, Idx, BestIdx, BestMatchTypeT,
                          StoredTypeT, StoredTypesT ...>
  : ParseVarValMatcher<
      ValueT, (ParseVarValScorer<ValueT, StoredTypeT>::Value > Score)
        ? ParseVarValScorer<ValueT, StoredTypeT>::Value
        : Score,
      Idx + 1, (ParseVarValScorer<ValueT, StoredTypeT>::Value > Score)
        ? Idx
        : BestIdx,
      typename std::conditional<
        (ParseVarValScorer<ValueT, StoredTypeT>::Value > Score),
        StoredTypeT,
        BestMatchTypeT>::type,
      StoredTypesT ...> {};

/// \brief Matcher for input value type for ParseVariant.
///
/// Returns the match object with:
/// - Value - indicating that match was found.
/// - Type -  matched type (field does not exist if match was not found).
/// - Index - index of matched type (or -1 if there is no match).
template <typename ValueT, typename ... StoredTypesT>
using ParseVarValMatcherT =
  ParseVarValMatcher<ValueT, 0, 0, 0, void, StoredTypesT ...>;

/// \brief SFINAE enabler for input value type for ParseVariant.
///
/// Works like enable_if.
template <typename ValueT, typename ... StoredTypesT>
using ParseVarValEnablerT =
  typename ParseVarValMatcherT<ValueT, StoredTypesT ...>::Type;


/// \brief Checks for standard is_ traits in stored types of ParseVariant.
// If there is no stored types return false as Value.
template <template <typename> class TraitT, typename ... StoredTypesT>
struct ParseVarTraitChecker {
  static const bool Value = false;
};

// Check for single stored type (one/last element case).
template <template <typename> class TraitT, typename StoredTypeT>
struct ParseVarTraitChecker<TraitT, StoredTypeT> {
  static const bool Value =
    TraitT<typename std::decay<StoredTypeT>::type>::value;
};

// Check for stored type (mutiple elements case).
template <template <typename> class TraitT,
          typename StoredTypeT, typename ... StoredTypesT>
struct ParseVarTraitChecker<TraitT, StoredTypeT, StoredTypesT ...> {
  static const bool Value =
    TraitT<typename std::decay<StoredTypeT>::type>::value &&
    ParseVarTraitChecker<TraitT, StoredTypesT ...>::Value;
};

/// \brief SFINAE enabler for copy constructor in ParseVariant.
template <typename ... StoredTypesT>
using ParseVarCopyEnablerT =
  typename std::enable_if<ParseVarTraitChecker<std::is_copy_constructible,
                                               StoredTypesT ...>::Value,
                          void>::type;

/// \brief SFINAE enabler for move constructor in ParseVariant.
template <typename ... StoredTypesT>
using ParseVarMoveEnablerT =
  typename std::enable_if<ParseVarTraitChecker<std::is_move_constructible,
                                               StoredTypesT ...>::Value,
                          void>::type;

/// \brief SFINAE enabler for default constructor in ParseVariant.
template <typename ... StoredTypesT>
using ParseVarDefCtorEnablerT =
  typename std::enable_if<ParseVarTraitChecker<std::is_default_constructible,
                                               StoredTypesT ...>::Value,
                          void>::type;


/// \brief Matcher for getAs<>(), unsafeGetAs<>() and isA<>() type parameter
///        for ParseVariant.
template <typename ValueT, typename ... StoredTypesT>
using ParseVarGetMatcherT =
  ParseVarValMatcherT<typename std::decay<ValueT>::type, StoredTypesT ...>;

/// \brief SFINAE enabler for getAs<>() and unsafeGetAs<>() type
///        parameter for ParseVariant.
template <typename ValueT, typename ... StoredTypesT>
using ParseVarGetEnablerT =
  typename ParseVarGetMatcherT<ValueT, StoredTypesT ...>::Type;


/// \brief Matcher for getAsExact<>(), unsafeGetAsExact<>() and isAExact<>()
///        type parameter for ParseVariant.
///
/// This is a raw helper, please use ParseVarGetExactMatcherT alias instead.
template <typename ValueT, bool IsMatch, int Idx, typename MatchedT,
          typename ... StoredTypesT>
struct ParseVarGetExactMatcher {
  static const bool Value = false;
  static const int Index = -1;
};

// Early finished (when match found).
template <typename ValueT, int Idx, typename MatchedT>
struct ParseVarGetExactMatcher<ValueT, true, Idx, MatchedT> {
  using Type = MatchedT;
  static const bool Value = true;
  static const int Index = Idx;
};

// Recursive scan of stored types.
template <typename ValueT, bool IsMatch, int Idx, typename MatchedT,
          typename StoredTypeT, typename ... StoredTypesT>
struct ParseVarGetExactMatcher<ValueT, IsMatch, Idx, MatchedT,
                               StoredTypeT, StoredTypesT ...>
  : std::conditional<
      std::is_same<ValueT, typename std::decay<StoredTypeT>::type>::value,
      ParseVarGetExactMatcher<ValueT, true, Idx, ValueT>,
      ParseVarGetExactMatcher<ValueT, false, Idx + 1, void, StoredTypesT ...>
    >::type {};

/// \brief Matcher for getAsExact<>(), unsafeGetAsExact<>() and isAExact<>()
///        type parameter for ParseVariant.
template <typename ValueT, typename ... StoredTypesT>
using ParseVarGetExactMatcherT = ParseVarGetExactMatcher<
  typename std::decay<ValueT>::type, false, 0, void, StoredTypesT ...>;

/// \brief SFINAE enabler for getAsExact<>() and unsafeGetAsExact<>()
///        type parameter for ParseVariant.
template <typename ValueT, typename ... StoredTypesT>
using ParseVarGetExactEnablerT = typename ParseVarGetExactMatcher<
  typename std::decay<ValueT>::type, false, 0, void, StoredTypesT ...>::Type;


/// \brief Value deleter for selected storage type in ParseVariant.
template <typename SelStorageT,
          bool = (std::is_class<SelStorageT>::value ||
                  std::is_union<SelStorageT>::value) &&
                 std::is_destructible<SelStorageT>::value>
struct ParseVarValDeleter {
  /// \brief Type of delete function which destroys object in variant.
  using DelFuncT = void (void *);


  static void destroy(void *Storage) {
    SelStorageT *SelStorage = static_cast<SelStorageT *>(Storage);
    if (SelStorage != nullptr)
      SelStorage->~SelStorageT();
  }


  static DelFuncT *getFunc() { return &destroy; }
};

template <typename SelStorageT>
struct ParseVarValDeleter<SelStorageT, false> {
  /// \brief Type of delete function which destroys object in variant.
  using DelFuncT = void (void *);


  static DelFuncT *getFunc() { return nullptr; }
};


/// \brief Value copier for selected storage type in ParseVariant.
template <typename SelStorageT,
          bool = std::is_copy_constructible<SelStorageT>::value>
struct ParseVarValCopier {
  /// \brief Type of copy function which copies object in variant.
  using CopyFuncT = void (void *, const void *);


  static void copy(void *Storage, const void *OtherStorage) {
    const SelStorageT *OtherSelStorage =
      static_cast<const SelStorageT *>(OtherStorage);
    if (Storage != nullptr && OtherSelStorage != nullptr)
      new (Storage) SelStorageT(*OtherSelStorage);
  }


  static CopyFuncT *getFunc() { return &copy; }
};

template <typename SelStorageT>
struct ParseVarValCopier<SelStorageT, false> {
  /// \brief Type of copy function which copies object in variant.
  using CopyFuncT = void (void *, const void *);


  static CopyFuncT *getFunc() { return nullptr; }
};


/// \brief Value mover for selected storage type in ParseVariant.
template <typename SelStorageT,
          bool = std::is_move_constructible<SelStorageT>::value,
          bool = std::is_copy_constructible<SelStorageT>::value>
struct ParseVarValMover {
  /// \brief Type of move function which moves object in variant.
  using MoveFuncT = void (void *, void *);


  static void move(void *Storage, void *OtherStorage) {
    SelStorageT *OtherSelStorage =
      static_cast<SelStorageT *>(OtherStorage);
    if (Storage != nullptr && OtherSelStorage != nullptr)
      new (Storage) SelStorageT(std::move(*OtherSelStorage));
  }


  static MoveFuncT *getFunc() { return &move; }
};

template <typename SelStorageT>
struct ParseVarValMover<SelStorageT, false, true> {
  /// \brief Type of move function which moves object in variant.
  using MoveFuncT = void (void *, void *);


  static void move(void *Storage, void *OtherStorage) {
    ParseVarValCopier<SelStorageT>::copy(Storage, OtherStorage);
  }


  static MoveFuncT *getFunc() { return &move; }
};

template <typename SelStorageT>
struct ParseVarValMover<SelStorageT, false, false> {
  /// \brief Type of move function which moves object in variant.
  using MoveFuncT = void (void *, void *);


  static MoveFuncT *getFunc() { return nullptr; }
};


// -----------------------------------------------------------------------------
// PARSE VARIANT IMPLEMENTATION
// -----------------------------------------------------------------------------

/// \brief Variant type that can represent many types.
///
/// \tparam StoredTypesT Types that can be represented by variant.
// TODO: The class should have conditionally disable the:
//        - default constructor
//        - copy constructor
//        - move constructor
//        - copy assignment operator
//        - move assignment operator
//        - destructor
//       if member type has corresponding special function inaccessible.
//       Current implementation will fail on that type:
//        - the SFINAE pattern is invalid for these members - no template
//          parameter
//        - these members cannot be template functions (part of them)
//       Need to add ParseVariantImpl with current functionality
//       and wrapper for it which will conditionally inherit set of empty
//       classes with locked/unlocked special functions. This will effectively
//       conditionally suppress creation of selected special functions in
//       wrapper.
//
//       The change is deferred though: The current issue does not affect
//       demangler code (all types are copy/move constructible) and new solution
//       will take time to implement (especially tricky move functions since
//       their can be suppressed by creation of other special functions and
//       some compilers like MSVC do not generate default at all - until 2015).
template <typename ... StoredTypesT>
class ParseVariant {
  /// \brief Type of storage for all elements.
  using StorageT = typename AlignedUnion<
    0, typename std::decay<StoredTypesT>::type ...>::type;
  /// \brief Type of delete function which destroys object in variant.
  using DelFuncT = void (void *);
  /// \brief Type of copy function which copies object in variant.
  using CopyFuncT = void (void *, const void *);
  /// \brief Type of move function which moves object in variant.
  using MoveFuncT = void (void *, void *);


public:
  /// Gets identifier (index in list of types) of current type.
  ///
  /// \return Index, or -1 if variant does not have value.
  int getStorageTypeId() const {
    return SelStorageIdx;
  }

  /// \brief Gets information whether input type can be currently stored in
  ///        current storage type (relaxed type matcher).
  template <typename ValueT>
  bool isA() const {
    using SelMatcherT = ParseVarGetMatcherT<ValueT, StoredTypesT ...>;

    return (SelMatcherT::Index >= 0 && SelMatcherT::Index == SelStorageIdx);
  }

  /// \brief Gets information whether input type is currently stored in
  ///        current storage type (strict type matcher).
  template <typename ValueT>
  bool isAExact() const {
    using SelMatcherT = ParseVarGetExactMatcherT<ValueT, StoredTypesT ...>;

    return (SelMatcherT::Index >= 0 && SelMatcherT::Index == SelStorageIdx);
  }

  /// \brief Gets variadic as specific type (relaxed type matcher).
  ///
  /// \return Pointer (of life-time of variant instance) of selected
  ///         stored type, or nullptr if variant does not have specified type.
  template <typename ValueT>
  const ParseVarGetEnablerT<ValueT, StoredTypesT ...> *getAs() const {
    // TODO: Move back to "using" once the bug in GCC 4.7 will be fixed:
    //       https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53540
    //using SelMatcherT = ParseVarGetMatcherT<ValueT, StoredTypesT ...>;
    typedef ParseVarGetMatcherT<ValueT, StoredTypesT ...> SelMatcherT;

    if (!isA<ValueT>())
      return nullptr;
    return reinterpret_cast<const typename SelMatcherT::Type *>(&Storage);
  }

  /// \brief Gets variadic as specific type (strict type matcher).
  ///
  /// \return Pointer (of life-time of variant instance) of selected
  ///         stored type, or nullptr if variant does not have specified type.
  template <typename ValueT>
  const ParseVarGetExactEnablerT<ValueT, StoredTypesT ...> *getAsExact() const {
    // TODO: Move back to "using" once the bug in GCC 4.7 will be fixed:
    //       https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53540
    //using SelMatcherT = ParseVarGetExactMatcherT<ValueT, StoredTypesT ...>;
    typedef ParseVarGetExactMatcherT<ValueT, StoredTypesT ...> SelMatcherT;

    if (!isAExact<ValueT>())
      return nullptr;
    return reinterpret_cast<const typename SelMatcherT::Type *>(&Storage);
  }

  /// \brief Gets variadic as specific type (unsafe, relaxed type matcher).
  ///
  /// \return Reference (of life-time of variant instance) of selected
  ///         stored type. Undefined behavior, if used with incorrect
  ///         ValueT (+ assert).
  template <typename ValueT>
  const ParseVarGetEnablerT<ValueT, StoredTypesT ...> &unsafeGetAs() const {
    // TODO: Move back to "using" once the bug in GCC 4.7 will be fixed:
    //       https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53540
    //using SelMatcherT = ParseVarGetMatcherT<ValueT, StoredTypesT ...>;
    typedef ParseVarGetMatcherT<ValueT, StoredTypesT ...> SelMatcherT;

    assert(isA<ValueT>() &&
           "Variant does not store specified type.");
    return reinterpret_cast<const typename SelMatcherT::Type &>(Storage);
  }

  /// \brief Gets variadic as specific type (unsafe, strict type matcher).
  ///
  /// \return Reference (of life-time of variant instance) of selected
  ///         stored type. Undefined behavior, if used with incorrect
  ///         ValueT (+ assert).
  template <typename ValueT>
  const ParseVarGetExactEnablerT<ValueT, StoredTypesT ...> &
  unsafeGetAsExact() const {
    // TODO: Move back to "using" once the bug in GCC 4.7 will be fixed:
    //       https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53540
    //using SelMatcherT = ParseVarGetExactMatcherT<ValueT, StoredTypesT ...>;
    typedef ParseVarGetExactMatcherT<ValueT, StoredTypesT ...> SelMatcherT;

    assert(isAExact<ValueT>() &&
           "Variant does not store specified type.");
    return reinterpret_cast<const typename SelMatcherT::Type &>(Storage);
  }


  /// \brief Creates new instance with no value.
  template <typename = ParseVarDefCtorEnablerT<StoredTypesT ...>>
  ParseVariant()
    : SelStorageIdx(-1), DelFunc(nullptr), CopyFunc(nullptr),
      MoveFunc(nullptr) {}

  /// \brief Creates new instance with one of possible values.
  template <typename ValueT,
            typename = ParseVarValEnablerT<ValueT, StoredTypesT ...>>
  ParseVariant(ValueT &&Val)
    : SelStorageIdx(ParseVarValMatcherT<ValueT, StoredTypesT ...>::Index) {
    using SelStorageT = ParseVarValEnablerT<ValueT, StoredTypesT ...>;

    new (static_cast<void *>(&Storage)) SelStorageT(std::forward<ValueT>(Val));
    DelFunc = ParseVarValDeleter<SelStorageT>::getFunc();
    CopyFunc = ParseVarValCopier<SelStorageT>::getFunc();
    MoveFunc = ParseVarValMover<SelStorageT>::getFunc();
  }

  /// \brief Copy constructor.
  ParseVariant(const ParseVariant &Other)
    : SelStorageIdx(Other.SelStorageIdx), DelFunc(Other.DelFunc),
      CopyFunc(Other.CopyFunc), MoveFunc(Other.MoveFunc) {
    if (CopyFunc != nullptr)
      CopyFunc(&Storage, &Other.Storage);
  }

  /// \brief Move constructor.
  ParseVariant(ParseVariant &&Other)
    : SelStorageIdx(Other.SelStorageIdx), DelFunc(Other.DelFunc),
      CopyFunc(Other.CopyFunc), MoveFunc(Other.MoveFunc) {
    if (MoveFunc != nullptr)
      MoveFunc(&Storage, &Other.Storage);
  }

  /// \brief Assigns one of possible values.
  template <typename ValueT,
            typename = ParseVarValEnablerT<ValueT, StoredTypesT ...>>
  ParseVariant &operator =(ValueT &&Val) {
    ParseVariant Tmp(std::forward<ValueT>(Val));
    assign(std::move(Tmp));
    return *this;
  }

  /// \brief Copy assignment operator.
  ParseVariant &operator =(const ParseVariant &Other) {
    ParseVariant Tmp(Other);
    assign(std::move(Tmp));
    return *this;
  }

  /// \brief Move assignment operator.
  ParseVariant &operator =(ParseVariant &&Other) {
    assign(std::move(Other));
    return *this;
  }

  /// \brief Destroys variant (calls proper destructor if necessary).
  ~ParseVariant() {
    clear();
  }

private:
  /// \brief Assigns variant to other variant (move object).
  ///
  /// Based on simple bitcopy of storage.
  void assign(ParseVariant &&Other) {
    if (this == &Other)
      return;

    clear();

    if (Other.MoveFunc != nullptr)
      Other.MoveFunc(&Storage, &Other.Storage);

    SelStorageIdx = Other.SelStorageIdx;
    DelFunc = Other.DelFunc;
    CopyFunc = Other.CopyFunc;
    MoveFunc = Other.MoveFunc;
  }

  /// \brief Clears variant state.
  void clear() {
    if (DelFunc != nullptr)
      DelFunc(&Storage);
  }


  /// Storage for variant type.
  StorageT Storage;
  /// Index of selected type in storage.
  int SelStorageIdx;
  /// Storage destructor function for current type.
  DelFuncT *DelFunc;
  /// Storage copy function for current type.
  CopyFuncT *CopyFunc;
  /// Storage move function for current type.
  MoveFuncT *MoveFunc;
};

} // adaptation
} // oclcxx

#endif // CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPARSEVARIANT_H
