// UNSUPPORTED: target=powerpc64-ibm-aix{{.*}}
// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/body1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/body2.c
// RUN: %clang_cc1 -emit-obj -o /dev/null -ast-merge %t.1.ast -ast-merge %t.2.ast %s
// expected-no-diagnostics
