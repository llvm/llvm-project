// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Full BMI: both EIDs survive.
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -I%t -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -ast-dump-all 2>&1 | FileCheck %s --check-prefix=FULL
//
// Reduced BMI: both EIDs survive (linked from the specialization).
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -I%t -emit-reduced-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -ast-dump-all 2>&1 | FileCheck %s --check-prefix=REDUCED

//--- header.h
#ifndef HEADER_H
#define HEADER_H
template <typename T>
struct GMFStruct {
  T value;
  T get() const { return value; }
};
#endif

//--- M.cppm
module;
#include "header.h"

// Explicit instantiation in GMF.
template struct GMFStruct<int>;

export module M;

export template <typename T>
struct PurvStruct {
  T value;
  T get() const { return value; }
};

// Explicit instantiation in module purview.
template struct PurvStruct<int>;

export using ::GMFStruct;

//--- use.cpp
// expected-no-diagnostics
import M;

// FULL: ExplicitInstantiationDecl {{.*}} imported in M.<global> {{.*}} explicit_instantiation_definition 'GMFStruct'
// FULL: ExplicitInstantiationDecl {{.*}} imported in M {{.*}} explicit_instantiation_definition 'PurvStruct'

// REDUCED: ExplicitInstantiationDecl {{.*}} imported in M.<global> {{.*}} explicit_instantiation_definition 'GMFStruct'
// REDUCED: ExplicitInstantiationDecl {{.*}} imported in M {{.*}} explicit_instantiation_definition 'PurvStruct'

void test() {
  GMFStruct<int> g;
  g.value = 1;

  PurvStruct<int> p;
  p.value = 2;
}
