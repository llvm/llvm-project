// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s

typedef int __attribute__((mode(byte))) __attribute__((vector_size(4))) vec_t1;
typedef int __attribute__((mode(QI))) __attribute__((vector_size(8))) vec_t2;
typedef int __attribute__((mode(SI))) __attribute__((vector_size(16))) vec_t3;
typedef int __attribute__((mode(DI))) __attribute__((vector_size(64)))vec_t4;
typedef float __attribute__((mode(SF))) __attribute__((vector_size(128))) vec_t5;
typedef float __attribute__((mode(DF))) __attribute__((vector_size(256))) vec_t6;

void check(void) {
  // CHECK: alloca <4 x i8>
  vec_t1 v1;
  // CHECK: alloca <8 x i8>
  vec_t2 v2;
  // CHECK: alloca <4 x i32>
  vec_t3 v3;
  // CHECK: alloca <8 x i64>
  vec_t4 v4;
  // CHECK: alloca <32 x float>
  vec_t5 v5;
  // CHECK: alloca <32 x double>
  vec_t6 v6;
}

// CHECK: ret i32 4
int check_size1(void) { return sizeof(vec_t1); }

// CHECK: ret i32 8
int check_size2(void) { return sizeof(vec_t2); }

// CHECK: ret i32 16
int check_size3(void) { return sizeof(vec_t3); }

// CHECK: ret i32 64
int check_size4(void) { return sizeof(vec_t4); }

// CHECK: ret i32 128
int check_size5(void) { return sizeof(vec_t5); }

// CHECK: ret i32 256
int check_size6(void) { return sizeof(vec_t6); }
