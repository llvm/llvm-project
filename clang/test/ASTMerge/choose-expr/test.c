// RUN: %clang_cc1 -std=c11 -emit-pch -o %t.ast %S/Inputs/choose.c
// RUN: %clang_cc1 -std=c11 -ast-merge %t.ast -fsyntax-only -verify %s
// expected-no-diagnostics

