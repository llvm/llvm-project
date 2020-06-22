// Test this without pch.
// RUN: %clang_cc1 -std=c++11 -include %S/cxx-variadic-templates.h -verify %s -ast-dump -o -
// RUN: %clang_cc1 -std=c++11 -include %S/cxx-variadic-templates.h %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -x c++-header -emit-pch -o %t %S/cxx-variadic-templates.h
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -include-pch %t %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -std=c++11 -x c++-header -emit-pch -fpch-instantiate-templates -o %t %S/cxx-variadic-templates.h
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++11 -include-pch %t %s -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: allocate_shared
shared_ptr<int> spi = shared_ptr<int>::allocate_shared(1, 2);

template<int> struct A {};
template<int> struct B {};
outer<int, int>::inner<1, 2, A, B> i(A<1>{}, B<2>{});
