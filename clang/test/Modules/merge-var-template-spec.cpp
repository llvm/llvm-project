// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// We need '-fmodules-local-submodule-visibility' to properly test merging when building a module from multiple
// headers inside the same TU. C++20 mode would imply this flag, but we need it to set it explicitly for C++14.
//
// RUN: %clang_cc1 -xc++ -std=c++14 -fmodules -fmodules-local-submodule-visibility -fmodule-name=library \
// RUN:     -emit-module %t/modules.map \
// RUN:     -o %t/module.pcm
//
//
// RUN: %clang_cc1 -xc++ -std=c++14 -fmodules -fmodules-local-submodule-visibility -fmodule-file=%t/module.pcm  \
// RUN:     -fmodule-map-file=%t/modules.map \
// RUN:     -fsyntax-only -verify %t/use.cpp
//
//--- use.cpp

#include "var1.h"
#include "var2.h"

auto foo = zero<Int>;
auto bar = zero<int*>;
auto baz = zero<int>;

template <class T> constexpr T zero = 0; // expected-error {{redefinition}} expected-note@* {{previous}}
template <> constexpr Int zero<Int> = {0}; // expected-error {{redefinition}} expected-note@* {{previous}}
template <class T> constexpr T* zero<T*> = nullptr; // expected-error {{redefinition}} expected-note@* {{previous}}

template <> constexpr int** zero<int**> = nullptr; // ok, new specialization.
template <class T> constexpr T** zero<T**> = nullptr; // ok, new partial specilization.

//--- modules.map
module "library" {
	export *
	module "var1" {
		export *
		header "var1.h"
	}
	module "var2" {
		export *
		header "var2.h"
	}
}

//--- var1.h
#ifndef VAR1_H
#define VAR1_H

template <class T> constexpr T zero = 0;
struct Int {
    int value;
};
template <> constexpr int zero<Int> = {0};
template <class T> constexpr T* zero<T*> = nullptr;

#endif // VAR1_H

//--- var2.h
#ifndef VAR2_H
#define VAR2_H

template <class T> constexpr T zero = 0;
struct Int {
    int value;
};
template <> constexpr int zero<Int> = {0};
template <class T> constexpr T* zero<T*> = nullptr;

#endif // VAR2_H
