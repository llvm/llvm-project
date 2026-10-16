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

void foo2(int v, int x) {
// CHECK: #pragma acc atomic read if(v)
// CHECK-NEXT:   v = x;
#pragma acc atomic read if (v)
  v = x;
// CHECK-NEXT: pragma acc atomic write if(x)
// CHECK-NEXT:  v = x + 1;
#pragma acc atomic write if (x)
  v = x + 1;
// CHECK-NEXT: pragma acc atomic update if(true)
// CHECK-NEXT:  x++;
#pragma acc atomic update if (true)
  x++;
// CHECK-NEXT: pragma acc atomic  if(false)
// CHECK-NEXT:  x--;
#pragma acc atomic if (false)
  x--;
// CHECK-NEXT: pragma acc atomic capture if(v < x)
// CHECK-NEXT:  v = x++;
#pragma acc atomic capture if (v < x)
  v = x++;

// CHECK-NEXT: #pragma acc atomic capture if(x > v)
// CHECK-NEXT: { 
// CHECK-NEXT: x--;
// CHECK-NEXT: v = x;
// CHECK-NEXT: }
#pragma acc atomic capture if (x > v)
  { x--; v = x; }

}
