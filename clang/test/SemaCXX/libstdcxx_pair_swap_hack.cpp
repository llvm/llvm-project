// This is a test for an egregious hack in Clang that works around
// an issue with GCC's <utility> implementation. std::pair::swap
// has an exception specification that makes an unqualified call to
// swap. This is invalid, because it ends up calling itself with
// the wrong number of arguments.
//
// The same problem afflicts a bunch of other class templates. Those
// affected are array, pair, priority_queue, stack, and queue.

// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=array
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=array -DPR28423
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=pair
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=priority_queue
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=stack
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=queue
//
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=array -DNAMESPACE=__debug
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=array -DNAMESPACE=__profile

// MSVC's standard library uses a very similar pattern that relies on delayed
// parsing of exception specifications.
//
// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify -fexceptions -fcxx-exceptions -DCLASS=array -DMSVC

#ifdef BE_THE_HEADER

#pragma GCC system_header
#ifdef PR28423
using namespace std;
#endif

namespace std {
  template<typename T> void swap(T &, T &);
  template<typename T> void do_swap(T &a, T &b) noexcept(noexcept(swap(a, b))) {
    swap(a, b);
  }

#ifdef NAMESPACE
  namespace NAMESPACE {
#define STD_CLASS std::NAMESPACE::CLASS
#else
#define STD_CLASS std::CLASS
#endif

  template<typename A, typename B> struct CLASS {
#ifdef MSVC
    void swap(CLASS &other) noexcept(noexcept(do_swap(member, other.member)));
#endif
    A member;
#ifndef MSVC
    void swap(CLASS &other) noexcept(noexcept(swap(member, other.member)));
#endif
  };

//  template<typename T> void do_swap(T &, T &);
//  template<typename A> struct vector {
//    void swap(vector &other) noexcept(noexcept(do_swap(member, other.member)));
//    A member;
//  };

#ifdef NAMESPACE
  }
#endif
}

#else

#define BE_THE_HEADER
#include __FILE__

struct X {};
using PX = STD_CLASS<X, X>;
using PI = STD_CLASS<int, int>;
void swap(X &, X &) noexcept;
PX px;
PI pi;

static_assert(noexcept(px.swap(px)), "");
static_assert(!noexcept(pi.swap(pi)), "");

namespace sad {
  template<typename T> void swap(T &, T &);

  template<typename A, typename B> struct CLASS {
    void swap(CLASS &other) noexcept(noexcept(swap(*this, other))); // expected-error {{too many arguments}} expected-note {{declared here}}
    // expected-error@-1{{uses itself}} expected-note@-1{{in instantiation of}}
  };

  CLASS<int, int> pi;

  static_assert(!noexcept(pi.swap(pi)), ""); // expected-note 2{{in instantiation of exception specification for 'swap'}}
}

#endif
