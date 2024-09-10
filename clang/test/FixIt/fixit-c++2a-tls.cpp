// RUN: %clang_cc1 -verify -std=c++2a -pedantic-errors %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -std=c++2a -fixit %t
// RUN: %clang_cc1 -Wall -pedantic-errors -x c++ -std=c++2a %t
// RUN: cat %t | FileCheck %s
// UNSUPPORTED: target={{.*-zos.*}}

/* This is a test of the various code modification hints that only
   apply in C++2a. */

namespace constinit_mismatch {
  extern thread_local constinit int a; // expected-note {{declared constinit here}}
  thread_local int a = 123; // expected-error {{'constinit' specifier missing on initializing declaration of 'a'}}
  // CHECK: {{^}}  constinit thread_local int a = 123;
}

