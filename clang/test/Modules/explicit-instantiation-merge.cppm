// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Two modules both include the same header and explicitly instantiate the same
// specialization. After importing both, the EIDs should be merged and at least
// one should survive.
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -I%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -ast-dump-all 2>&1 | FileCheck %s --check-prefix=CHECK-AST
//
// Verify cross-module duplicate diagnostic points to the module's EID location.
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/dup.cpp -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -I%t -emit-reduced-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -ast-dump-all 2>&1 | FileCheck %s --check-prefix=CHECK-AST
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/dup.cpp -verify -fsyntax-only

//--- header.h
#ifndef HEADER_H
#define HEADER_H
template <typename T>
struct S {
  T value;
};
#endif

//--- A.cppm
module;
#include "header.h"
template struct S<int>; // #A-inst
export module A;
export using ::S;

//--- B.cppm
module;
#include "header.h"
template struct S<int>;
export module B;
export using ::S;

//--- use.cpp
// expected-no-diagnostics
import A;
import B;

// CHECK-AST: ExplicitInstantiationDecl {{.*}} imported {{.*}} explicit_instantiation_definition 'S'

void test() {
  S<int> s;
  s.value = 42;
}

//--- dup.cpp
import A;

template struct S<int>; // expected-error {{duplicate explicit instantiation of 'S<int>'}}
                        // expected-note@A.cppm:3 {{previous explicit instantiation is here}}
