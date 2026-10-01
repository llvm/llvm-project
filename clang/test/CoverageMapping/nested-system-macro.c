// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c11 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name nested-system-macro.c %s | FileCheck %s

// A system macro nested in a user macro must stay in the enclosing macro's
// virtual file. In particular, do not introduce a disconnected zero-length
// code region at the invocation of a macro which defines a function.
#include "Inputs/nested-system-macro/user.h"
typedef struct object { int value; } object;
DECLARE_CLASS(object)

void direct(void) {
  USER_NOP();
}

int main(void) {
  object o = {0};
  return object_cast(&o)->value;
}

// CHECK-LABEL: direct:
// CHECK-NEXT: File 0, 10:19 -> 12:2 = #0
// CHECK-NEXT: Expansion,File 0, 11:3 -> 11:11 = #0 (Expanded file = 1)
// CHECK: Branch,File 1, 2:33 -> 2:45 = 0, #0

// CHECK-LABEL: nested-system-macro.c:object_cast:
// CHECK-NEXT: File 0, 6:38 -> 9:4 = #0
// CHECK-NEXT: Expansion,File 0, 7:5 -> 7:16 = #0 (Expanded file = 1)
// CHECK-NEXT: File 1, 4:27 -> 4:43 = #0
// CHECK-NEXT: Expansion,File 1, 4:27 -> 4:41 = #0 (Expanded file = 2)
// CHECK: Branch,File 2, 3:39 -> 3:57 = 0, #0
