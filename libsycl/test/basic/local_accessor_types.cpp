//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -Wno-error=deprecated-declarations %s -fsyntax-only \
// RUN: -Xclang -verify -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

template <int Dims, typename AccT> struct FullSubscriptType;
template <typename AccT> struct FullSubscriptType<1, AccT> {
  using type = decltype(std::declval<AccT>()[0]);
};
template <typename AccT> struct FullSubscriptType<2, AccT> {
  using type = decltype(std::declval<AccT>()[0][0]);
};
template <typename AccT> struct FullSubscriptType<3, AccT> {
  using type = decltype(std::declval<AccT>()[0][0][0]);
};

template <typename DataT, int Dims, typename AccT> void checkTypes() {
  static_assert(std::is_same_v<typename AccT::value_type, DataT>);
  static_assert(std::is_same_v<typename AccT::reference, DataT &>);
  static_assert(std::is_same_v<typename AccT::const_reference, const DataT &>);
  static_assert(std::is_same_v<typename AccT::size_type, std::size_t>);
  static_assert(std::is_same_v<typename AccT::difference_type,
                               typename std::iterator_traits<
                                   typename AccT::iterator>::difference_type>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().begin()),
                               typename AccT::iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().cbegin()),
                               typename AccT::const_iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().rbegin()),
                               typename AccT::reverse_iterator>);
  static_assert(std::is_same_v<decltype(std::declval<AccT>().crbegin()),
                               typename AccT::const_reverse_iterator>);

  if constexpr (Dims == 0) {
    static_assert(std::is_convertible_v<const AccT &, DataT &>);
  } else {
    static_assert(
        std::is_same_v<typename FullSubscriptType<Dims, AccT>::type, DataT &>);
    static_assert(
        std::is_same_v<decltype(std::declval<AccT>()[sycl::id<Dims>{}]),
                       DataT &>);
    static_assert(std::is_same_v<decltype(std::declval<AccT>().get_range()),
                                 sycl::range<Dims>>);
  }
}

template <typename DataT> void checkAllDims() {
  checkTypes<DataT, 0, sycl::local_accessor<DataT, 0>>();
  checkTypes<DataT, 1, sycl::local_accessor<DataT, 1>>();
  checkTypes<DataT, 2, sycl::local_accessor<DataT, 2>>();
  checkTypes<DataT, 3, sycl::local_accessor<DataT, 3>>();
}

template <int Dims> void checkDeprecatedAccessor() {
  // expected-warning@+1 {{'local' is deprecated: use 'local_accessor' instead}}
  constexpr sycl::target Local = sycl::target::local;
  using AccT = sycl::accessor<int, Dims, sycl::access_mode::read_write, Local>;
  checkTypes<int, Dims, AccT>();
  static_assert(
      std::is_same_v<decltype(std::declval<AccT>().get_size()), std::size_t>);
  static_assert(
      std::is_same_v<decltype(std::declval<AccT>().get_count()), std::size_t>);
}

void checkAll() {
  checkAllDims<int>();
  checkAllDims<const int>();
  checkAllDims<double>();

  checkDeprecatedAccessor<0>();
  checkDeprecatedAccessor<1>();
  checkDeprecatedAccessor<2>();
  checkDeprecatedAccessor<3>();

  static_assert(std::is_convertible_v<sycl::local_accessor<int, 2>,
                                      sycl::local_accessor<const int, 2>>);
  static_assert(!std::is_convertible_v<sycl::local_accessor<const int, 1>,
                                       sycl::local_accessor<int, 1>>);
  static_assert(!std::is_convertible_v<sycl::local_accessor<int, 1>,
                                       sycl::local_accessor<int, 2>>);
  static_assert(std::is_default_constructible_v<sycl::local_accessor<int, 1>>);
  static_assert(
      !std::is_constructible_v<sycl::local_accessor<int, 1>, sycl::handler &>);
}

// The SYCL 1.2.1 spellings of the template arguments name the same type.
// expected-warning@+4 {{'mode' is deprecated: use 'sycl::access_mode' instead}}
// expected-warning@+3 {{'target' is deprecated: use 'sycl::target' instead}}
// expected-warning@+2 {{'local' is deprecated: use 'local_accessor' instead}}
using LegacyAccT = sycl::accessor<
    int, 2, sycl::access::mode::read_write, sycl::access::target::local,
    // expected-warning@+2 {{'placeholder' is deprecated: placeholder accessors are deprecated in SYCL 2020}}
    // expected-warning@+1 {{'false_t' is deprecated: placeholder accessors are deprecated in SYCL 2020}}
    sycl::access::placeholder::false_t>;
// expected-warning@+1 {{'local' is deprecated: use 'local_accessor' instead}}
constexpr sycl::target LocalTarget = sycl::target::local;

template <int Dims>
using DeprecatedAccT =
    sycl::accessor<int, Dims, sycl::access_mode::read_write, LocalTarget>;
static_assert(std::is_same_v<LegacyAccT, DeprecatedAccT<2>>);

// SYCL 2020 defines no properties for a local accessor.
static_assert(!sycl::is_property_of_v<sycl::property::queue::in_order,
                                      sycl::local_accessor<int, 1>>);
static_assert(!sycl::is_property_of_v<sycl::property::queue::in_order,
                                      DeprecatedAccT<1>>);

// Element access in a kernel yields generic references, although the
// accessors point to local memory there.
void checkTypesInKernel(sycl::handler &CGH) {
  sycl::local_accessor<int, 1> Local(1, CGH);
  sycl::local_accessor<const int, 2> ConstLocal({1, 1}, CGH);
  DeprecatedAccT<0> Deprecated0(CGH);
  DeprecatedAccT<1> Deprecated1(1, CGH);
  DeprecatedAccT<3> Deprecated3({1, 1, 1}, CGH);
  CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
    static_assert(std::is_same_v<decltype(Local[0]), int &>);
    static_assert(std::is_same_v<decltype(*Local.begin()), int &>);
    static_assert(std::is_same_v<decltype(ConstLocal[0][0]), const int &>);
    static_assert(
        std::is_same_v<decltype(ConstLocal[sycl::id<2>{}]), const int &>);
    static_assert(std::is_convertible_v<decltype(Deprecated0), int &>);
    static_assert(std::is_same_v<decltype(Deprecated1[0]), int &>);
    static_assert(std::is_same_v<decltype(Deprecated3[0][0][0]), int &>);
    static_assert(std::is_same_v<decltype(Deprecated3[sycl::id<3>{}]), int &>);
  });
}

void checkMaxSizeDeprecated(const sycl::local_accessor<int, 1> &Acc) {
  // expected-warning@+1 {{'max_size' is deprecated: max_size() is deprecated by SYCL 2020}}
  (void)Acc.max_size();
}
