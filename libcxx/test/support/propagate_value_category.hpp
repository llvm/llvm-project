//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_PROPAGATE_VALUE_CATEGORY
#define TEST_SUPPORT_PROPAGATE_VALUE_CATEGORY

#include "test_macros.h"
#include <type_traits>

#if TEST_STD_VER < 11
#error this header may only be used in C++11
#endif

using UnderlyingVCType = unsigned;
enum ValueCategory : UnderlyingVCType {
  VC_None = 0,
  VC_LVal = 1 << 0,
  VC_RVal = 1 << 1,
  VC_Const = 1 << 2,
  VC_Volatile = 1 << 3,
  VC_ConstVolatile = VC_Const | VC_Volatile
};

inline constexpr ValueCategory operator&(ValueCategory LHS, ValueCategory RHS) {
  return ValueCategory(LHS & (UnderlyingVCType)RHS);
}

inline constexpr ValueCategory operator|(ValueCategory LHS, ValueCategory RHS) {
  return ValueCategory(LHS | (UnderlyingVCType)RHS);
}

inline constexpr ValueCategory operator^(ValueCategory LHS, ValueCategory RHS) {
  return ValueCategory(LHS ^ (UnderlyingVCType)RHS);
}

inline constexpr bool isValidValueCategory(ValueCategory VC) {
  return (VC & (VC_LVal | VC_RVal)) != (VC_LVal | VC_RVal);
}

inline constexpr bool hasValueCategory(ValueCategory Arg, ValueCategory Key) {
  return Arg == Key || ((Arg & Key) == Key);
}

template <class Tp>
using UnCVRef =
    typename std::remove_cv<typename std::remove_reference<Tp>::type>::type;

template <class Tp>
constexpr ValueCategory getReferenceQuals() {
  return std::is_lvalue_reference<Tp>::value
             ? VC_LVal
             : (std::is_rvalue_reference<Tp>::value ? VC_RVal : VC_None);
}
static_assert(getReferenceQuals<int>() == VC_None, "");
static_assert(getReferenceQuals<int &>() == VC_LVal, "");
static_assert(getReferenceQuals<int &&>() == VC_RVal, "");

template <class Tp>
constexpr ValueCategory getCVQuals() {
  using Vp = typename std::remove_reference<Tp>::type;
  return std::is_const<Vp>::value && std::is_volatile<Vp>::value
             ? VC_ConstVolatile
             : (std::is_const<Vp>::value
                    ? VC_Const
                    : (std::is_volatile<Vp>::value ? VC_Volatile : VC_None));
}
static_assert(getCVQuals<int>() == VC_None, "");
static_assert(getCVQuals<int const>() == VC_Const, "");
static_assert(getCVQuals<int volatile>() == VC_Volatile, "");
static_assert(getCVQuals<int const volatile>() == VC_ConstVolatile, "");
static_assert(getCVQuals<int &>() == VC_None, "");
static_assert(getCVQuals<int const &>() == VC_Const, "");

template <class Tp>
inline constexpr ValueCategory getValueCategory() {
  return getReferenceQuals<Tp>() | getCVQuals<Tp>();
}
static_assert(getValueCategory<int>() == VC_None, "");
static_assert(getValueCategory<int const &>() == (VC_LVal | VC_Const), "");
static_assert(getValueCategory<int const volatile &&>() ==
                  (VC_RVal | VC_ConstVolatile),
              "");

template <ValueCategory VC>
struct ApplyValueCategory {
private:
  static_assert(isValidValueCategory(VC), "");

  template <bool Pred, class Then, class Else>
  using CondT = typename std::conditional<Pred, Then, Else>::type;

public:
  template <class Tp, class Vp = UnCVRef<Tp>>
  using ApplyCVQuals = CondT<
      hasValueCategory(VC, VC_ConstVolatile), typename std::add_cv<Vp>::type,
      CondT<hasValueCategory(VC, VC_Const), typename std::add_const<Vp>::type,
            CondT<hasValueCategory(VC, VC_Volatile),
                  typename std::add_volatile<Vp>::type, Tp>>>;

  template <class Tp, class Vp = typename std::remove_reference<Tp>::type>
  using ApplyReferenceQuals =
      CondT<hasValueCategory(VC, VC_LVal),
            typename std::add_lvalue_reference<Vp>::type,
            CondT<hasValueCategory(VC, VC_RVal),
                  typename std::add_rvalue_reference<Vp>::type, Vp>>;

  template <class Tp>
  using Apply = ApplyReferenceQuals<ApplyCVQuals<UnCVRef<Tp>>>;

  template <class Tp, bool Dummy = true,
            typename std::enable_if<Dummy && (VC & VC_LVal), bool>::type = true>
  static Apply<UnCVRef<Tp>> cast(Tp &&t) {
    using ToType = Apply<UnCVRef<Tp>>;
    return static_cast<ToType>(t);
  }

  template <class Tp, bool Dummy = true,
            typename std::enable_if<Dummy && (VC & VC_RVal), bool>::type = true>
  static Apply<UnCVRef<Tp>> cast(Tp &&t) {
    using ToType = Apply<UnCVRef<Tp>>;
    return static_cast<ToType>(std::move(t));
  }

  template <
      class Tp, bool Dummy = true,
      typename std::enable_if<Dummy && ((VC & (VC_LVal | VC_RVal)) == VC_None),
                              bool>::type = true>
  static Apply<UnCVRef<Tp>> cast(Tp &&t) {
    return t;
  }
};

template <ValueCategory VC, class Tp>
using ApplyValueCategoryT = typename ApplyValueCategory<VC>::template Apply<Tp>;

template <class Tp>
using PropagateValueCategory = ApplyValueCategory<getValueCategory<Tp>()>;

template <class Tp, class Up>
using PropagateValueCategoryT =
    typename ApplyValueCategory<getValueCategory<Tp>()>::template Apply<Up>;

template <ValueCategory VC, class Tp>
typename ApplyValueCategory<VC>::template Apply<Tp> ValueCategoryCast(Tp &&t) {
  return ApplyValueCategory<VC>::cast(std::forward<Tp>(t));
};

#endif // TEST_SUPPORT_PROPAGATE_VALUE_CATEGORY
