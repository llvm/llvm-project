// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -emit-llvm -O2 -o - -triple hexagon-unknown-elf -target-cpu hexagonv60 %s | FileCheck %s

// The return value should return the value in A[1].
// Check that the HexagonBuiltinExpr doesn't evaluate &(*ptr++) twice. If so,
// the return value will be the value in A[2]
// CHECK: @brev_ptr_inc
// CHECK-DAG: llvm.hexagon.L2.loadri.pbr
// CHECK-DAG: getelementptr inbounds nuw i8, {{.*}}i32 4
// CHECK-NOT: getelementptr inbounds nuw i8, {{.*}}i32 8
// CHECK-NOT: getelementptr inbounds nuw i8, {{.*}}i32 4
int brev_ptr_inc(int A[], int B[]) {
  int *p0 = &B[0];
  int *p1 = &A[0];
  __builtin_brev_ldw(p0, &*p1++, 8);
  return (*p1);
}

// The return value should return the value in A[0].
// CHECK: @brev_ptr_dec
// CHECK: llvm.hexagon.L2.loadri.pbr
// CHECK: [[RET:%[0-9]+]] = load{{.*}}%A
// CHECK: ret{{.*}}[[RET]]
int brev_ptr_dec(int A[], int B[]) {
  int *p0 = &B[0];
  int *p1 = &A[1];
  __builtin_brev_ldw(p0, &*p1--, 8);
  return (*p1);
}

// The store in bitcode needs to be of width correspondng to 16-bit.
// CHECK: @brev_ptr_half
// CHECK: llvm.hexagon.L2.loadrh.pbr
// CHECK: store{{.*}}i16{{.*}}ptr
short int brev_ptr_half(short int A[], short int B[]) {
  short int *p0 = &B[0];
  short int *p1 = &A[0];
  __builtin_brev_ldh(p0, &*p1++, 8);
  return (*p1);
}

// The store in bitcode needs to be of width correspondng to 8-bit.
// CHECK: @brev_ptr_byte
// CHECK: llvm.hexagon.L2.loadrub.pbr
// CHECK: store{{.*}}i8{{.*}}ptr
unsigned char brev_ptr_byte(unsigned char A[], unsigned char B[]) {
  unsigned char *p0 = &B[0];
  unsigned char *p1 = &A[0];
  __builtin_brev_ldub(p0, &*p1++, 8);
  return (*p1);
}

