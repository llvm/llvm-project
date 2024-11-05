// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/class-template-spec.cpp
// RUN: %clang_cc1 -ast-merge %t.1.ast -fsyntax-only -verify %s
// expected-no-diagnostics

template struct N0::A<short>;
template struct N1::A<short>;
template struct N2::A<short>;
template struct N3::B<short>;
