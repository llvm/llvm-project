// RUN: %clang_cc1 -triple i686-pc-linux -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -triple i686-pc-linux -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux -fexperimental-new-constant-interpreter -verify=warn %s
// RUN: %clang_cc1 -triple x86_64-pc-linux -verify=warn %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -verify %s
// RUN: %clang_cc1 -triple hexagon -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -triple hexagon -verify %s

// expected-no-diagnostics

/// This test is split out from the rest since the output is target dependent.

enum E { // warn-warning {{enumeration values exceed range of largest integer}}
  E1 = -__LONG_MAX__ -1L,
  E2 = __LONG_MAX__ *2UL+1UL
};

