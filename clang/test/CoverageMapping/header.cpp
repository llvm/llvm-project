// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name header.cpp %s > %tmapping
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-FUNC
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-STATIC-FUNC
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-STATIC-FUNC2
//
// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -mllvm -limited-coverage-experimental=true -dump-coverage-mapping -emit-llvm-only -main-file-name header.cpp %s > %tmapping.limited
// RUN: FileCheck -input-file %tmapping.limited %s --check-prefix=CHECK-LIMITED

#include "Inputs/header1.h"

int main() {
  func(1);
  static_func(2);
}

// CHECK-FUNC: func
// CHECK-FUNC: File 0, 4:25 -> 11:2 = #0
// CHECK-FUNC: File 0, 6:15 -> 8:4 = #1
// CHECK-FUNC: File 0, 8:10 -> 10:4 = (#0 - #1)

// CHECK-STATIC-FUNC: static_func
// CHECK-STATIC-FUNC: File 0, 12:32 -> 20:2 = #0
// CHECK-STATIC-FUNC: File 0, 14:15 -> 16:4 = #1
// CHECK-STATIC-FUNC: File 0, 16:10 -> 18:4 = (#0 - #1)

// CHECK-STATIC-FUNC2: static_func2
// CHECK-STATIC-FUNC2: File 0, 21:33 -> 29:2 = 0

// CHECK-LIMITED-NOT: static_func2
