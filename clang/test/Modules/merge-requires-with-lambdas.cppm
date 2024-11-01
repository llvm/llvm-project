// Tests that we can merge the concept declarations with lambda well.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/A0.cppm -emit-module-interface -o %t/A0.pcm
// RUN: %clang_cc1 -std=c++20 %t/TestA.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/A1.cppm -emit-module-interface -o %t/A1.pcm
// RUN: %clang_cc1 -std=c++20 %t/TestA1.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/A2.cppm -emit-module-interface -o %t/A2.pcm
// RUN: %clang_cc1 -std=c++20 %t/TestA2.cpp -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/A3.cppm -emit-module-interface -o %t/A3.pcm
// RUN: %clang_cc1 -std=c++20 %t/TestA3.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- A.h
template <class _Tp>
concept A = requires(const _Tp& __t) { []<class __Up>(const __Up&) {}(__t); };

//--- A1.h
template <class _Tp>
concept A = requires(const _Tp& __t) { []<class __Up>(__Up) {}(__t); };

//--- A2.h
template <class _Tp>
concept A = requires(const _Tp& __t) { []<class __Up>(const __Up& __u) {
    (int)__u;
}(__t); };

//--- A3.h
template <class _Tp>
concept A = requires(const _Tp& __t) { [t = '?']<class __Up>(const __Up&) {
    (int)t;
}(__t); };

//--- A.cppm
module;
#include "A.h"
export module A;
export using ::A;

//--- A0.cppm
module;
#include "A.h"
export module A0;
export using ::A;

//--- TestA.cpp
// expected-no-diagnostics
import A;
import A0;

template <class C>
void f(C) requires(A<C>) {}

//--- A1.cppm
module;
#include "A1.h"
export module A1;
export using ::A;

//--- TestA1.cpp
import A;
import A1;

template <class C>
void f(C) requires(A<C>) {} // expected-error 1+{{reference to 'A' is ambiguous}}
                            // expected-note@* 1+{{candidate found by name lookup is 'A'}}

//--- A2.cppm
module;
#include "A2.h"
export module A2;
export using ::A;

//--- TestA2.cpp
import A;
import A2;

template <class C>
void f(C) requires(A<C>) {} // expected-error 1+{{reference to 'A' is ambiguous}}
                            // expected-note@* 1+{{candidate found by name lookup is 'A'}}

//--- A3.cppm
module;
#include "A3.h"
export module A3;
export using ::A;

//--- TestA3.cpp
import A;
import A3;

template <class C>
void f(C) requires(A<C>) {} // expected-error 1+{{reference to 'A' is ambiguous}}
                            // expected-note@* 1+{{candidate found by name lookup is 'A'}}
