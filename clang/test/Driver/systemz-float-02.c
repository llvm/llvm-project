// RUN: %clang -target s390x-linux-gnu -march=z13 -S %s -o - -msoft-float | FileCheck %s
// REQUIRES: systemz-registered-target

// Check that -msoft-float works all the way to assembly output.

double fun0(double *A) {
// CHECK-LABEL: fun0
// CHECK-NOT: {{%f[0-9]}}
// CHECK: brasl %r14, __adddf3@PLT
  return *A + 1.0;
}

typedef int v4si __attribute__ ((vector_size (16)));
v4si fun1(v4si *A) {
// CHECK-LABEL: fun1
// CHECK-NOT: {{%[v][0-9]}}
// CHECK: ark
// CHECK-NEXT: ark
// CHECK-NEXT: ark
// CHECK-NEXT: ark
  v4si B = {1, 1, 1, 1};
  return *A + B;
}
