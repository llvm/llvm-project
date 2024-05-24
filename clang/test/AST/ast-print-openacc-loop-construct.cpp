// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

void foo() {
// CHECK: #pragma acc loop
// CHECK-NEXT: for (;;)
// CHECK-NEXT: ;
#pragma acc loop
  for(;;);
}
