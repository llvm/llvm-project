// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-traits.h -std=c++11 -fsyntax-only -verify %s

// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -o %t %S/cxx-traits.h
// RUN: %clang_cc1 -std=c++11 -include-pch %t -fsyntax-only -verify %s

// expected-no-diagnostics

bool _Is_pod_comparator = __is_pod<int>::__value;
bool _Is_empty_check = __is_empty<int>::__value;

bool default_construct_int = is_trivially_constructible<int>::value;
bool copy_construct_int = is_trivially_constructible<int, const int&>::value;
