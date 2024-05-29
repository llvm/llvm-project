// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify=good -Wno-tentative-definition-array %s
// good-no-diagnostics

int foo[]; // expected-warning {{tentative array definition assumed to have one element}}
