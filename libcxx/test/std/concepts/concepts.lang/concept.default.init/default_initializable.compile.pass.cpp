//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// We voluntarily use std::default_initializable on types that have redundant
// or ignored cv-qualifiers -- don't warn about it.
// ADDITIONAL_COMPILE_FLAGS: -Wno-ignored-qualifiers

// template<class T>
//     concept default_initializable = constructible_from<T> &&
//     requires { T{}; } &&
//     is-default-initializable<T>;

#include <array>
#include <concepts>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <span>
#include <stack>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_macros.h"

struct Empty {};

struct CtorDefaulted {
  CtorDefaulted() = default;
};
struct CtorDeleted {
  CtorDeleted() = delete;
};
struct DtorDefaulted {
  ~DtorDefaulted() = default;
};
struct DtorDeleted {
  ~DtorDeleted() = delete;
};

struct Noexcept {
  ~Noexcept() noexcept;
};
struct NoexceptTrue {
  ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse {
  ~NoexceptFalse() noexcept(false);
};

struct CtorProtected {
protected:
  CtorProtected() = default;
};
struct CtorPrivate {
private:
  CtorPrivate() = default;
};
struct DtorProtected {
protected:
  ~DtorProtected() = default;
};
struct DtorPrivate {
private:
  ~DtorPrivate() = default;
};

template <class T>
struct NoexceptDependant {
  ~NoexceptDependant() noexcept(std::is_same_v<T, int>);
};

struct CtorExplicit {
  explicit CtorExplicit() = default;
};
struct CtorArgument {
  CtorArgument(int) {}
};
struct CtorDefaultArgument {
  CtorDefaultArgument(int = 0) {}
};
struct CtorExplicitDefaultArgument {
  explicit CtorExplicitDefaultArgument(int = 0) {}
};

struct Derived : public Empty {};

class Abstract {
  virtual void foo() = 0;
};

class AbstractDestructor {
  virtual ~AbstractDestructor() = 0;
};

class OperatorNewDeleted {
  void* operator new(std::size_t) = delete;
  void operator delete(void* ptr) = delete;
};

[[maybe_unused]] auto Lambda = [](const int&, int&&, double){};

template<class T>
void test_not_const()
{
    static_assert( std::default_initializable<               T>);
    static_assert(!std::default_initializable<const          T>);
    static_assert( std::default_initializable<      volatile T>);
    static_assert(!std::default_initializable<const volatile T>);
}

template<class T>
void test_true()
{
    static_assert( std::default_initializable<               T>);
    static_assert( std::default_initializable<const          T>);
    static_assert( std::default_initializable<      volatile T>);
    static_assert( std::default_initializable<const volatile T>);
}

template<class T>
void test_false()
{
    static_assert(!std::default_initializable<               T>);
    static_assert(!std::default_initializable<const          T>);
    static_assert(!std::default_initializable<      volatile T>);
    static_assert(!std::default_initializable<const volatile T>);
}

void test()
{
    test_not_const<bool>();
    test_not_const<char>();
    test_not_const<int>();
    test_not_const<double>();

    test_false    <void>();
    test_not_const<void*>();

    test_not_const<int*>();
    test_false    <int[]>();
    test_not_const<int[1]>();
    test_false    <int&>();
    test_false    <int&&>();

    test_true     <Empty>();

    test_true     <CtorDefaulted>();
    test_false    <CtorDeleted>();
    test_true     <DtorDefaulted>();
    test_false    <DtorDeleted>();

    test_true     <Noexcept>();
    test_true     <NoexceptTrue>();
    test_false    <NoexceptFalse>();

    test_false    <CtorProtected>();
    test_false    <CtorPrivate>();
    test_false    <DtorProtected>();
    test_false    <DtorPrivate>();

    test_true     <NoexceptDependant<int>>();
    test_false    <NoexceptDependant<double>>();

    test_true     <CtorExplicit>();
    test_false    <CtorArgument>();
    test_true     <CtorDefaultArgument>();
    test_true     <CtorExplicitDefaultArgument>();

    test_true     <Derived>();
    test_false    <Abstract>();
    test_false    <AbstractDestructor>();

    test_true     <OperatorNewDeleted>();

    test_true     <decltype(Lambda)>();
    test_not_const<void(*)(const int&)>();
    test_not_const<void(Empty::*)(const int&)               >();
    test_not_const<void(Empty::*)(const int&) const         >();
    test_not_const<void(Empty::*)(const int&)       volatile>();
    test_not_const<void(Empty::*)(const int&) const volatile>();
    test_not_const<void(Empty::*)(const int&) &>();
    test_not_const<void(Empty::*)(const int&) &&>();
    test_not_const<void(Empty::*)(const int&) noexcept>();
    test_not_const<void(Empty::*)(const int&) noexcept(true)>();
    test_not_const<void(Empty::*)(const int&) noexcept(false)>();

    // Sequence containers
    test_not_const<std::array<               int, 0>>();
    test_not_const<std::array<               int, 1>>();
    test_false    <std::array<const          int, 1>>();
    test_not_const<std::array<      volatile int, 1>>();
    test_false    <std::array<const volatile int, 1>>();
    test_true     <std::deque<               int>>();
#ifdef _LIBCPP_VERSION
    test_true     <std::deque<const          int>>();
#endif // _LIBCPP_VERSION
    test_true     <std::forward_list<int>>();
    test_true     <std::list<int>>();
    test_true     <std::vector<int>>();

    // Associative containers
    test_true     <std::set<int>>();
    test_true     <std::map<int, int>>();
    test_true     <std::multiset<int>>();
    test_true     <std::multimap<int, int>>();

    // Unordered associative containers
    test_true     <std::unordered_set<int>>();
    test_true     <std::unordered_map<int, int>>();
    test_true     <std::unordered_multiset<int>>();
    test_true     <std::unordered_multimap<int, int>>();

    // Container adaptors
    test_true     <std::stack<               int>>();
#ifdef _LIBCPP_VERSION
    test_true     <std::stack<const          int>>();
#endif // _LIBCPP_VERSION
    test_true     <std::queue<int>>();
    test_true     <std::priority_queue<int>>();

    test_true     <std::span<               int>>();
    test_true     <std::span<const          int>>();
    test_true     <std::span<      volatile int>>();
    test_true     <std::span<const volatile int>>();

    // Strings
    test_true     <std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_true     <std::wstring>();
#endif
    test_true     <std::u8string>();
    test_true     <std::u16string>();
    test_true     <std::u32string>();

    // String views
    test_true     <std::string_view>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test_true     <std::wstring_view>();
#endif
    test_true     <std::u8string_view>();
    test_true     <std::u16string_view>();
    test_true     <std::u32string_view>();

    // Smart pointers
    test_true     <std::unique_ptr<int>>();
    test_true     <std::shared_ptr<int>>();
    test_true     <std::weak_ptr<int>>();

}
