// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/test.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- header.h
#pragma once
template <class _Tp>
class Optional {};

template <class _Tp>
concept C = requires(const _Tp& __t) {
    []<class _Up>(const Optional<_Up>&) {}(__t);
};

//--- func.h
#include "header.h"
template <C T>
void func() {}

//--- test_func.h
#include "func.h"

inline void test_func() {
    func<Optional<int>>();
}

//--- A.cppm
module;
#include "header.h"
#include "test_func.h"
export module A;
export using ::test_func;

//--- test.cpp
// expected-no-diagnostics
import A;
#include "test_func.h"

void test() {
    test_func();
}
