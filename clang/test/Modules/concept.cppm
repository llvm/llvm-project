// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t -DDIFFERENT %t/B.cppm -verify
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/B.cppm -verify
//
// Testing the behavior of `-fskip-odr-check-in-gmf`
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf -fprebuilt-module-path=%t -I%t  \
// RUN:    -DDIFFERENT -DSKIP_ODR_CHECK_IN_GMF %t/B.cppm -verify


//--- foo.h
#ifndef FOO_H
#define FOO_H

template <class T>
concept Range = requires(T &t) { t.begin(); };

template<class _Tp>
concept __integer_like = true;

template <class _Tp>
concept __member_size = requires(_Tp &&t) { t.size(); };

template <class First, class Second>
concept C = requires(First x, Second y) { x + y; };

struct A {
public:
  template <Range T>
  using range_type = T;
};

struct __fn {
  template <__member_size _Tp>
  constexpr __integer_like auto operator()(_Tp&& __t) const {
    return __t.size();
  }

  template <__integer_like _Tp, C<_Tp> Sentinel>
  constexpr _Tp operator()(_Tp &&__t, Sentinel &&last) const {
    return __t;
  }

  template <template <class> class H, class S, C<H<S>> Sentinel>
  constexpr H<S> operator()(H<S> &&__s, Sentinel &&last) const {
    return __s;
  }

// Tests that we could find different concept definition indeed.
#ifndef DIFFERENT
  template <__integer_like _Tp, __integer_like _Up, C<_Tp> Sentinel>
  constexpr _Tp operator()(_Tp &&__t, _Up _u, Sentinel &&last) const {
    return __t;
  }
#else
  template <__integer_like _Tp, __integer_like _Up, C<_Up> Sentinel>
  constexpr _Tp operator()(_Tp &&__t, _Up _u, Sentinel &&last) const {
    return __t;
  }
#endif
};
#endif

//--- A.cppm
module;
#include "foo.h"
export module A;

//--- B.cppm
module;
#include "foo.h"
export module B;
import A;

#ifdef SKIP_ODR_CHECK_IN_GMF
// expected-error@B.cppm:* {{call to object of type '__fn' is ambiguous}}
// expected-note@* 1+{{candidate function}}
#elif defined(DIFFERENT)
// expected-error@foo.h:41 {{'__fn::operator()' from module 'A.<global>' is not present in definition of '__fn' provided earlier}}
// expected-note@* 1+{{declaration of 'operator()' does not match}}
#else
// expected-no-diagnostics
#endif

template <class T>
struct U {
  auto operator+(U) { return 0; }
};

void foo() {
    A a;
    struct S {
        int size() { return 0; }
        auto operator+(S s) { return 0; }
    };
    __fn{}(S());
    __fn{}(S(), S());
    __fn{}(S(), S(), S());

    __fn{}(U<int>(), U<int>());
}
