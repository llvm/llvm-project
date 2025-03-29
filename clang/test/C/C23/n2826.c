// RUN: %clang_cc1 -ffreestanding -emit-llvm -o - -std=c2x %s | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -std=c17 -verify %s

/* WG14 N2826: Clang 17
 * Add annotations for unreachable control flow v2
 */
#include <stddef.h>

enum E {
  Zero,
  One,
  Two,
};

int test(enum E e) {
  switch (e) {
  case Zero: return 0;
  case One: return 1;
  case Two: return 2;
  }
  unreachable(); // expected-error {{call to undeclared function 'unreachable'}}
}

// CHECK: switch i32 %0, label %[[EPILOG:.+]] [
// CHECK: [[EPILOG]]:
// CHECK-NEXT: unreachable
