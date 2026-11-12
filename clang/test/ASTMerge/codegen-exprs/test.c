// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-pch -o %t.1.ast %S/Inputs/exprs1.c
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-pch -o %t.2.ast %S/Inputs/exprs2.c
// RUN: %clang_cc1 -triple %itanium_abi_triple -o /dev/null -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only -verify %s
// expected-no-diagnostics

