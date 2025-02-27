// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

void foo(int v, int x) {
// CHECK: #pragma acc atomic read
// CHECK-NEXT:   v = x;
#pragma acc atomic read
  v = x;
// CHECK-NEXT: pragma acc atomic write
// CHECK-NEXT:  v = x + 1;
#pragma acc atomic write
  v = x + 1;
// CHECK-NEXT: pragma acc atomic update
// CHECK-NEXT:  x++;
#pragma acc atomic update
  x++;
// CHECK-NEXT: pragma acc atomic 
// CHECK-NEXT:  x--;
#pragma acc atomic
  x--;
// CHECK-NEXT: pragma acc atomic capture
// CHECK-NEXT:  v = x++;
#pragma acc atomic capture
  v = x++;

// CHECK-NEXT: #pragma acc atomic capture
// CHECK-NEXT: { 
// CHECK-NEXT: x--;
// CHECK-NEXT: v = x;
// CHECK-NEXT: }
#pragma acc atomic capture
  { x--; v = x; }

}
