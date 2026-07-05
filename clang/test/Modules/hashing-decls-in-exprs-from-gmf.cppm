// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fskip-odr-check-in-gmf %t/B.cppm -emit-module-interface -o %t/B.pcm
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

//--- duplicated_func.h
#include "header.h"
template <C T>
void duplicated_func() {}

//--- test_func.h
#include "func.h"

void test_func() {
    func<Optional<int>>();
}

//--- test_duplicated_func.h
#include "duplicated_func.h"

void test_duplicated_func() {
    duplicated_func<Optional<int>>();
}

//--- A.cppm
module;
#include "header.h"
#include "test_duplicated_func.h"
export module A;
export using ::test_duplicated_func;

//--- B.cppm
module;
#include "header.h"
#include "test_func.h"
#include "test_duplicated_func.h"
export module B;
export using ::test_func;
export using ::test_duplicated_func;

//--- test.cpp
// expected-no-diagnostics
import A;
import B;

void test() {
    test_func();
    test_duplicated_func();
}
