// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -fixit %t
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Werror %t

struct a {
  int x; //< comment // expected-warning {{not a Doxygen trailing comment}}
  int y; /*< comment */ // expected-warning {{not a Doxygen trailing comment}}
};

// CHECK: fix-it:"{{.*}}":{8:10-8:13}:"///<"
// CHECK: fix-it:"{{.*}}":{9:10-9:13}:"/**<"

