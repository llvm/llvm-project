// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fwrapv | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -ftrapv | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fwrapv-pointer | FileCheck %s --check-prefix=FWRAPV-POINTER

void test(void) {
  // -fwrapv-pointer should turn off inbounds for GEP's
  extern int* P;
  ++P;
  // DEFAULT: getelementptr inbounds nuw i32, ptr
  // FWRAPV-POINTER: getelementptr i32, ptr
}

struct S {
  int a;
  int b;
  int c: 10;
  int d: 10;
};

int test_struct(struct S* s) {
  // -fwrapv-pointer should turn off inbounds nuw for struct GEP's
  return s->b;
  // DEFAULT: getelementptr inbounds nuw %struct.S, ptr
  // FWRAPV-POINTER: getelementptr %struct.S, ptr
}

int test_struct_bitfield(struct S* s) {
  // -fwrapv-pointer should turn off inbounds nuw for struct GEP's
  return s->d;
  // DEFAULT: getelementptr inbounds nuw %struct.S, ptr
  // FWRAPV-POINTER: getelementptr %struct.S, ptr
}
