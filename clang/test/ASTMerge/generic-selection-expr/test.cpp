// RUN: %clang_cc1 -std=c++03 -emit-pch -o %t.ast %S/Inputs/generic.cpp
// RUN: %clang_cc1 -std=c++03 -ast-merge %t.ast -fsyntax-only -verify %s
// expected-no-diagnostics
