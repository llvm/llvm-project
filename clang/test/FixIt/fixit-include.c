// RUN: %clang_cc1 -fsyntax-only -Wall -pedantic -verify %s
// RUN: mkdir -p %t-dir
// RUN: cp %s %t-dir/fixit-include.c
// RUN: cp %S/fixit-include.h %t-dir/fixit-include.h
// RUN: not %clang_cc1 -fixit %t-dir/fixit-include.c
// RUN: %clang_cc1 -Wall -pedantic %t-dir/fixit-include.c
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <fixit-include.h> // expected-error {{'fixit-include.h' file not found with <angled> include; use "quotes" instead}}
// CHECK: fix-it:{{.*}}:{9:10-9:27}

#pragma does_not_exist // expected-warning {{unknown pragma ignored}}

int main( void ) {
  return 0;
}
