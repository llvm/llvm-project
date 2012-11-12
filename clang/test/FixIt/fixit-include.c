// RUN: %clang_cc1 -fsyntax-only -Wall -pedantic -verify %s
// RUN: cp %s %t
// RUN: cp %S/fixit-include.h %T
// RUN: not %clang_cc1 -fsyntax-only -fixit %t
// RUN: %clang_cc1 -Wall -pedantic %t
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <fixit-include.h> // expected-error {{'fixit-include.h' file not found with <angled> include; use "quotes" instead}}
// CHECK: fix-it:{{.*}}:{8:10-8:27}

#pragma does_not_exist // expected-warning {{unknown pragma ignored}}

int main( void ) {
  return 0;
}
