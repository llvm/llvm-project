//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class From, class To>
// concept assignable_from;

#include <concepts>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../support/allocators.h"

// Note: is_lvalue_reference is checked in all ModelsAssignableFrom calls.
template <typename T1, typename T2>
constexpr void NeverAssignableFrom() {
  static_assert(!std::assignable_from<T1, T2>);
  static_assert(!std::assignable_from<T1, const T2>);
  static_assert(!std::assignable_from<T1, T2&>);
  static_assert(!std::assignable_from<T1, const T2&>);
  static_assert(!std::assignable_from<T1, volatile T2&>);
  static_assert(!std::assignable_from<T1, const volatile T2&>);
  static_assert(!std::assignable_from<T1, T2&&>);
  static_assert(!std::assignable_from<T1, const T2&&>);
  static_assert(!std::assignable_from<T1, volatile T2&&>);
  static_assert(!std::assignable_from<T1, const volatile T2&&>);

  static_assert(!std::assignable_from<const volatile T1&, T2>);
  static_assert(!std::assignable_from<const volatile T1&, const T2>);
  static_assert(!std::assignable_from<const volatile T1&, T2&>);
  static_assert(!std::assignable_from<const volatile T1&, const T2&>);
  static_assert(!std::assignable_from<const volatile T1&, volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&, const volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&, T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, const T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, volatile T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, const volatile T2&&>);

  static_assert(!std::assignable_from<T1&&, T2>);
  static_assert(!std::assignable_from<T1&&, const T2>);
  static_assert(!std::assignable_from<T1&&, T2&>);
  static_assert(!std::assignable_from<T1&&, const T2&>);
  static_assert(!std::assignable_from<T1&&, volatile T2&>);
  static_assert(!std::assignable_from<T1&&, const volatile T2&>);
  static_assert(!std::assignable_from<T1&&, T2&&>);
  static_assert(!std::assignable_from<T1&&, const T2&&>);
  static_assert(!std::assignable_from<T1&&, volatile T2&&>);
  static_assert(!std::assignable_from<T1&&, const volatile T2&&>);

  static_assert(!std::assignable_from<const T1&&, T2>);
  static_assert(!std::assignable_from<const T1&&, const T2>);
  static_assert(!std::assignable_from<const T1&&, T2&>);
  static_assert(!std::assignable_from<const T1&&, const T2&>);
  static_assert(!std::assignable_from<const T1&&, volatile T2&>);
  static_assert(!std::assignable_from<const T1&&, const volatile T2&>);
  static_assert(!std::assignable_from<const T1&&, T2&&>);
  static_assert(!std::assignable_from<const T1&&, const T2&&>);
  static_assert(!std::assignable_from<const T1&&, volatile T2&&>);
  static_assert(!std::assignable_from<const T1&&, const volatile T2&&>);

  static_assert(!std::assignable_from<volatile T1&&, T2>);
  static_assert(!std::assignable_from<volatile T1&&, const T2>);
  static_assert(!std::assignable_from<volatile T1&&, T2&>);
  static_assert(!std::assignable_from<volatile T1&&, const T2&>);
  static_assert(!std::assignable_from<volatile T1&&, volatile T2&>);
  static_assert(!std::assignable_from<volatile T1&&, const volatile T2&>);
  static_assert(!std::assignable_from<volatile T1&&, T2&&>);
  static_assert(!std::assignable_from<volatile T1&&, const T2&&>);
  static_assert(!std::assignable_from<volatile T1&&, volatile T2&&>);
  static_assert(!std::assignable_from<volatile T1&&, const volatile T2&&>);

  static_assert(!std::assignable_from<const volatile T1&&, T2>);
  static_assert(!std::assignable_from<const volatile T1&&, const T2>);
  static_assert(!std::assignable_from<const volatile T1&&, T2&>);
  static_assert(!std::assignable_from<const volatile T1&&, const T2&>);
  static_assert(!std::assignable_from<const volatile T1&&, volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&&, const volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&&, T2&&>);
  static_assert(!std::assignable_from<const volatile T1&&, const T2&&>);
  static_assert(!std::assignable_from<const volatile T1&&, volatile T2&&>);
  static_assert(
      !std::assignable_from<const volatile T1&&, const volatile T2&&>);

  static_assert(!std::assignable_from<const T1&, T2>);
  static_assert(!std::assignable_from<const T1&, const T2>);
  static_assert(!std::assignable_from<const T1&, T2&>);
  static_assert(!std::assignable_from<const T1&, const T2&>);
  static_assert(!std::assignable_from<const T1&, volatile T2&>);
  static_assert(!std::assignable_from<const T1&, const volatile T2&>);
  static_assert(!std::assignable_from<const T1&, T2&&>);
  static_assert(!std::assignable_from<const T1&, const T2&&>);
  static_assert(!std::assignable_from<const T1&, volatile T2&&>);
  static_assert(!std::assignable_from<const T1&, const volatile T2&&>);

  static_assert(!std::assignable_from<const volatile T1&, T2>);
  static_assert(!std::assignable_from<const volatile T1&, const T2>);
  static_assert(!std::assignable_from<const volatile T1&, T2&>);
  static_assert(!std::assignable_from<const volatile T1&, const T2&>);
  static_assert(!std::assignable_from<const volatile T1&, volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&, const volatile T2&>);
  static_assert(!std::assignable_from<const volatile T1&, T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, const T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, volatile T2&&>);
  static_assert(!std::assignable_from<const volatile T1&, const volatile T2&&>);
}

template <typename T1, typename T2>
constexpr bool CheckAssignableFromRvalues() {
  NeverAssignableFrom<T1, T2>();

  constexpr auto Result = std::assignable_from<T1&, T2>;
  static_assert(std::assignable_from<T1&, T2&&> == Result);

  return Result;
}

template <typename T1, typename T2>
constexpr bool CheckAssignableFromLvalues() {
  NeverAssignableFrom<T1, T2>();

  constexpr auto Result = std::assignable_from<T1&, const T2&>;
  static_assert(std::assignable_from<T1&, T2&> == Result);
  static_assert(std::assignable_from<T1&, const T2&> == Result);

  return Result;
}

template <typename T1, typename T2>
constexpr bool CheckAssignableFromLvaluesAndRvalues() {
  return CheckAssignableFromLvalues<T1, T2>() &&
         CheckAssignableFromRvalues<T1, T2>();
}

namespace BuiltinTypes {
static_assert(CheckAssignableFromLvaluesAndRvalues<int, int>());
static_assert(CheckAssignableFromLvaluesAndRvalues<int, double>());
static_assert(CheckAssignableFromLvaluesAndRvalues<double, int>());
static_assert(CheckAssignableFromLvaluesAndRvalues<int*, int*>());
static_assert(!CheckAssignableFromLvaluesAndRvalues<int*, const int*>());
static_assert(!CheckAssignableFromLvaluesAndRvalues<int*, volatile int*>());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<int*, const volatile int*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<const int*, int*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<const int*, const int*>());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<const int*, volatile int*>());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<const int*, const volatile int*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<volatile int*, int*>());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<volatile int*, const int*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<volatile int*, volatile int*>());
static_assert(!CheckAssignableFromLvaluesAndRvalues<volatile int*,
                                                    const volatile int*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<const volatile int*, int*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<const volatile int*, const int*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<const volatile int*, volatile int*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<const volatile int*,
                                                   const volatile int*>());

static_assert(CheckAssignableFromLvaluesAndRvalues<int (*)(), int (*)()>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<int (*)(), int (*)() noexcept>());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<int (*)() noexcept, int (*)()>());

struct S {};
static_assert(CheckAssignableFromLvaluesAndRvalues<int S::*, int S::*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<const int S::*, int S::*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<int (S::*)(), int (S::*)()>());
static_assert(CheckAssignableFromLvaluesAndRvalues<int (S::*)(),
                                                   int (S::*)() noexcept>());
static_assert(CheckAssignableFromLvaluesAndRvalues<int (S::*)() const,
                                                   int (S::*)() const>());
static_assert(CheckAssignableFromLvaluesAndRvalues<
              int (S::*)() const, int (S::*)() const noexcept>());
static_assert(CheckAssignableFromLvaluesAndRvalues<int (S::*)() volatile,
                                                   int (S::*)() volatile>());
static_assert(CheckAssignableFromLvaluesAndRvalues<
              int (S::*)() volatile, int (S::*)() volatile noexcept>());
static_assert(CheckAssignableFromLvaluesAndRvalues<
              int (S::*)() const volatile, int (S::*)() const volatile>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<
        int (S::*)() const volatile, int (S::*)() const volatile noexcept>());

static_assert(!std::assignable_from<void, int>);
static_assert(!std::assignable_from<void, void>);
static_assert(!std::assignable_from<int*&, long*>);
static_assert(!std::assignable_from<int (&)[5], int[5]>);
static_assert(!std::assignable_from<int (S::*&)(), int (S::*)() const>);
static_assert(!std::assignable_from<int (S::*&)() const, int (S::*)()>);
static_assert(!std::assignable_from<int (S::*&)(), int (S::*)() volatile>);
static_assert(!std::assignable_from<int (S::*&)() volatile, int (S::*)()>);
static_assert(
    !std::assignable_from<int (S::*&)(), int (S::*)() const volatile>);
static_assert(
    !std::assignable_from<int (S::*&)() const volatile, int (S::*)()>);
static_assert(!std::assignable_from<int (S::*&)() noexcept, int (S::*)()>);
static_assert(
    !std::assignable_from<int (S::*&)() const noexcept, int (S::*)() const>);
static_assert(!std::assignable_from<int (S::*&)() volatile noexcept,
                                    int (S::*)() volatile>);
static_assert(!std::assignable_from<int (S::*&)() const volatile noexcept,
                                    int (S::*)() const volatile>);
} // namespace BuiltinTypes

namespace TypesFitForPurpose {
struct T1 {};

struct NoCommonReference {};
static_assert(!std::common_reference_with<const T1&, const NoCommonReference&>);
static_assert(!std::assignable_from<NoCommonReference&, T1>);

struct AssignmentReturnsNonReference {
  AssignmentReturnsNonReference operator=(T1);
  operator T1() const;
};
static_assert(std::common_reference_with<const T1&,
                                         const AssignmentReturnsNonReference&>);
static_assert(!std::assignable_from<AssignmentReturnsNonReference&, T1>);

struct NonCVAssignmentOnly {
  NonCVAssignmentOnly& operator=(T1);
  operator T1() const;
};
static_assert(
    std::common_reference_with<const T1&, const NonCVAssignmentOnly&>);
static_assert(std::assignable_from<NonCVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<const NonCVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<volatile NonCVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<const volatile NonCVAssignmentOnly&, T1>);

struct NonCVAssignmentOnlyConstQualified {
  NonCVAssignmentOnlyConstQualified& operator=(T1) const;
  operator T1() const;
};
static_assert(std::common_reference_with<
              const T1&, const NonCVAssignmentOnlyConstQualified&>);
static_assert(std::assignable_from<NonCVAssignmentOnlyConstQualified&, T1>);
static_assert(
    !std::assignable_from<const NonCVAssignmentOnlyConstQualified&, T1>);
static_assert(
    !std::assignable_from<volatile NonCVAssignmentOnlyConstQualified&, T1>);
static_assert(!std::assignable_from<
              const volatile NonCVAssignmentOnlyConstQualified&, T1>);

struct NonCVAssignmentVolatileQualified {
  NonCVAssignmentVolatileQualified& operator=(T1) volatile;
  operator T1() const volatile;
};
static_assert(std::common_reference_with<
              const T1&, const NonCVAssignmentVolatileQualified&>);
static_assert(std::assignable_from<NonCVAssignmentVolatileQualified&, T1>);
static_assert(
    !std::assignable_from<const NonCVAssignmentVolatileQualified&, T1>);
static_assert(
    !std::assignable_from<volatile NonCVAssignmentVolatileQualified&, T1>);
static_assert(!std::assignable_from<
              const volatile NonCVAssignmentVolatileQualified&, T1>);

struct NonCVAssignmentOnlyCVQualified {
  NonCVAssignmentOnlyCVQualified& operator=(T1) const volatile;
  operator T1() const volatile;
};
static_assert(std::common_reference_with<
              const T1&, const NonCVAssignmentOnlyCVQualified&>);
static_assert(std::assignable_from<NonCVAssignmentOnlyCVQualified&, T1>);
static_assert(!std::assignable_from<const NonCVAssignmentOnlyCVQualified&, T1>);
static_assert(
    !std::assignable_from<volatile NonCVAssignmentOnlyCVQualified&, T1>);
static_assert(
    !std::assignable_from<const volatile NonCVAssignmentOnlyCVQualified&, T1>);

struct ConstAssignmentOnly {
  const ConstAssignmentOnly& operator=(T1) const;
  operator T1() const;
};
static_assert(
    std::common_reference_with<const T1&, const ConstAssignmentOnly&>);
static_assert(std::assignable_from<const ConstAssignmentOnly&, T1>);
static_assert(!std::assignable_from<ConstAssignmentOnly&, T1>);
static_assert(!std::assignable_from<volatile ConstAssignmentOnly&, T1>);
static_assert(!std::assignable_from<const volatile ConstAssignmentOnly&, T1>);

struct VolatileAssignmentOnly {
  volatile VolatileAssignmentOnly& operator=(T1) volatile;
  operator T1() const volatile;
};
static_assert(
    std::common_reference_with<const T1&, const VolatileAssignmentOnly&>);
static_assert(!std::assignable_from<VolatileAssignmentOnly&, T1>);
static_assert(std::assignable_from<volatile VolatileAssignmentOnly&, T1>);

struct CVAssignmentOnly {
  const volatile CVAssignmentOnly& operator=(T1) const volatile;
  operator T1() const volatile;
};
static_assert(std::common_reference_with<const T1&, const CVAssignmentOnly&>);
static_assert(std::assignable_from<const volatile CVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<CVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<const CVAssignmentOnly&, T1>);
static_assert(!std::assignable_from<volatile CVAssignmentOnly&, T1>);

struct LvalueRefQualifiedWithRvalueT1Only {
  LvalueRefQualifiedWithRvalueT1Only& operator=(T1&&) &;
  const LvalueRefQualifiedWithRvalueT1Only& operator=(T1&&) const&;
  volatile LvalueRefQualifiedWithRvalueT1Only& operator=(T1&&) volatile&;
  const volatile LvalueRefQualifiedWithRvalueT1Only& operator=(T1&&) const
      volatile&;
  operator T1() const volatile;
};
static_assert(std::common_reference_with<
              const T1&, const LvalueRefQualifiedWithRvalueT1Only&>);
static_assert(std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, T1&&>);
static_assert(
    std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&, T1&&>);
static_assert(
    std::assignable_from<volatile LvalueRefQualifiedWithRvalueT1Only&, T1&&>);
static_assert(std::assignable_from<
              const volatile LvalueRefQualifiedWithRvalueT1Only&, T1&&>);
static_assert(!std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, T1&>);
static_assert(
    !std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&, T1&>);
static_assert(
    !std::assignable_from<volatile LvalueRefQualifiedWithRvalueT1Only&, T1&>);
static_assert(!std::assignable_from<
              const volatile LvalueRefQualifiedWithRvalueT1Only&, T1&>);
static_assert(
    !std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, const T1&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    const T1&>);
static_assert(!std::assignable_from<
              volatile LvalueRefQualifiedWithRvalueT1Only&, const T1&>);
static_assert(!std::assignable_from<
              const volatile LvalueRefQualifiedWithRvalueT1Only&, const T1&>);
static_assert(
    !std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, volatile T1&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    volatile T1&>);
static_assert(!std::assignable_from<
              volatile LvalueRefQualifiedWithRvalueT1Only&, volatile T1&>);
static_assert(
    !std::assignable_from<const volatile LvalueRefQualifiedWithRvalueT1Only&,
                          volatile T1&>);
static_assert(!std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&,
                                    const volatile T1&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    const volatile T1&>);
static_assert(
    !std::assignable_from<volatile LvalueRefQualifiedWithRvalueT1Only&,
                          const volatile T1&>);
static_assert(
    !std::assignable_from<const volatile LvalueRefQualifiedWithRvalueT1Only&,
                          const volatile T1&>);
static_assert(
    !std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, const T1&&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    const T1&&>);
static_assert(!std::assignable_from<
              volatile LvalueRefQualifiedWithRvalueT1Only&, const T1&&>);
static_assert(!std::assignable_from<
              const volatile LvalueRefQualifiedWithRvalueT1Only&, const T1&&>);
static_assert(
    !std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&, volatile T1&&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    volatile T1&&>);
static_assert(!std::assignable_from<
              volatile LvalueRefQualifiedWithRvalueT1Only&, volatile T1&&>);
static_assert(
    !std::assignable_from<const volatile LvalueRefQualifiedWithRvalueT1Only&,
                          volatile T1&&>);
static_assert(!std::assignable_from<LvalueRefQualifiedWithRvalueT1Only&,
                                    const volatile T1&&>);
static_assert(!std::assignable_from<const LvalueRefQualifiedWithRvalueT1Only&,
                                    const volatile T1&&>);
static_assert(
    !std::assignable_from<volatile LvalueRefQualifiedWithRvalueT1Only&,
                          const volatile T1&&>);
static_assert(
    !std::assignable_from<const volatile LvalueRefQualifiedWithRvalueT1Only&,
                          const volatile T1&&>);

struct NoLvalueRefAssignment {
  NoLvalueRefAssignment& operator=(T1) &&;
  const NoLvalueRefAssignment& operator=(T1) const&&;
  volatile NoLvalueRefAssignment& operator=(T1) volatile&&;
  const volatile NoLvalueRefAssignment& operator=(T1) const volatile&&;
  operator T1() const volatile;
};
static_assert(
    std::common_reference_with<const T1&, const NoLvalueRefAssignment&>);
static_assert(!std::assignable_from<NoLvalueRefAssignment&, T1>);
static_assert(!std::assignable_from<NoLvalueRefAssignment&, const T1>);
static_assert(!std::assignable_from<NoLvalueRefAssignment&, volatile T1>);
static_assert(!std::assignable_from<NoLvalueRefAssignment&, const volatile T1>);
static_assert(!std::assignable_from<const NoLvalueRefAssignment&, T1>);
static_assert(!std::assignable_from<const NoLvalueRefAssignment&, const T1>);
static_assert(!std::assignable_from<const NoLvalueRefAssignment&, volatile T1>);
static_assert(
    !std::assignable_from<const NoLvalueRefAssignment&, const volatile T1>);
static_assert(!std::assignable_from<volatile NoLvalueRefAssignment&, T1>);
static_assert(!std::assignable_from<volatile NoLvalueRefAssignment&, const T1>);
static_assert(
    !std::assignable_from<volatile NoLvalueRefAssignment&, volatile T1>);
static_assert(
    !std::assignable_from<volatile NoLvalueRefAssignment&, const volatile T1>);
static_assert(!std::assignable_from<const volatile NoLvalueRefAssignment&, T1>);
static_assert(
    !std::assignable_from<const volatile NoLvalueRefAssignment&, const T1>);
static_assert(
    !std::assignable_from<const volatile NoLvalueRefAssignment&, volatile T1>);
static_assert(!std::assignable_from<const volatile NoLvalueRefAssignment&,
                                    const volatile T1>);
} // namespace TypesFitForPurpose

namespace StandardTypes {
static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::deque<int>, std::deque<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::deque<int>,
                                                    std::deque<const int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::deque<int>, std::deque<int, A1<int> > >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::deque<int>,
                                                    std::vector<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::deque<int>, int>());

static_assert(CheckAssignableFromLvaluesAndRvalues<std::forward_list<int>,
                                                   std::forward_list<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::forward_list<int>, std::forward_list<const int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::forward_list<int>, std::forward_list<int, A1<int> > >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::forward_list<int>,
                                                    std::vector<int> >());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<std::forward_list<int>, int>());

static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::list<int>, std::list<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::list<int>,
                                                    std::list<const int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::list<int>, std::list<int, A1<int> > >());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<std::list<int>, std::vector<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::list<int>, int>());

static_assert(CheckAssignableFromLvaluesAndRvalues<std::map<int, void*>,
                                                   std::map<int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::map<int, void*>, std::map<const int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::map<int, void*>,
              std::map<int, void*, A1<std::pair<int, void*> > > >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::map<int, void*>, std::unordered_map<int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::map<int, void*>,
                                                    std::pair<int, void*> >());

#ifndef _LIBCPP_HAS_NO_THREADS
static_assert(!CheckAssignableFromRvalues<std::mutex, std::mutex>());
static_assert(!CheckAssignableFromLvalues<std::mutex, std::mutex>());
#endif

static_assert(CheckAssignableFromLvaluesAndRvalues<std::optional<int>, int>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::optional<int>, double>());
static_assert(CheckAssignableFromLvaluesAndRvalues<std::optional<int>,
                                                   std::optional<int> >());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<int, std::optional<int> >());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<double, std::optional<int> >());

static_assert(
    !std::common_reference_with<std::optional<int>, std::optional<double> >);
static_assert(
    !CheckAssignableFromRvalues<std::optional<int>, std::optional<double> >());
static_assert(
    !CheckAssignableFromLvalues<std::optional<int>, std::optional<double> >());

static_assert(CheckAssignableFromLvaluesAndRvalues<std::string, std::string>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::string, std::string_view>());
static_assert(CheckAssignableFromLvaluesAndRvalues<std::string, char*>());
static_assert(CheckAssignableFromLvaluesAndRvalues<std::string, const char*>());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::string, std::basic_string<wchar_t> >());
static_assert(
    !CheckAssignableFromLvaluesAndRvalues<std::string, std::vector<char> >());

static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::string_view, std::string_view>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::string_view, std::string>());
static_assert(CheckAssignableFromLvaluesAndRvalues<std::string_view, char*>());
static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::string_view, const char*>());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::string_view, std::basic_string_view<wchar_t> >());

static_assert(
    CheckAssignableFromRvalues<std::unique_ptr<int>, std::unique_ptr<int> >());
static_assert(
    !CheckAssignableFromLvalues<std::unique_ptr<int>, std::unique_ptr<int> >());

static_assert(
    CheckAssignableFromLvaluesAndRvalues<std::unordered_map<int, void*>,
                                         std::unordered_map<int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::unordered_map<int, void*>,
              std::unordered_map<const int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::unordered_map<int, void*>,
              std::unordered_map<int, void*, A1<std::pair<int, void*> > > >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::unordered_map<int, void*>, std::map<int, void*> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::unordered_map<int, void*>, std::pair<int, void*> >());

static_assert(CheckAssignableFromLvaluesAndRvalues<std::vector<int>,
                                                   std::vector<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::vector<int>,
                                                    std::vector<const int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<
              std::vector<int>, std::vector<int, A1<int> > >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::vector<int>,
                                                    std::deque<int> >());
static_assert(!CheckAssignableFromLvaluesAndRvalues<std::vector<int>, int>());
} // namespace StandardTypes

int main(int, char**) { return 0; }
