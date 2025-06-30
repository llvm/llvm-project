// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck -input-file=%t-cir.ll %s

void foo() {
  unsigned long b = sizeof(bool);
  // CHECK: store i64 1, ptr {{%.*}}, align 8

  unsigned long i = sizeof(int);
  // CHECK: store i64 4, ptr {{%.*}}, align 8

  unsigned long l =  sizeof(long);
  // CHECK: store i64 8, ptr {{%.*}}, align 8

  unsigned long f =  sizeof(float);
  // CHECK: store i64 4, ptr {{%.*}}, align 8

  unsigned long d =  sizeof(double);
  // CHECK: store i64 8, ptr {{%.*}}, align 8

  unsigned long iArr =  sizeof(float[5]);
  // CHECK: store i64 20, ptr {{%.*}}, align 8

  unsigned long dArr =  sizeof(double[5]);
  // CHECK: store i64 40, ptr {{%.*}}, align 8
}

void foo2() {
  unsigned long b = alignof(bool);
  // CHECK: store i64 1, ptr {{%.*}}, align 8

  unsigned long i = alignof(int);
  // CHECK: store i64 4, ptr {{%.*}}, align 8

  unsigned long l =  alignof(long);
  // CHECK: store i64 8, ptr {{%.*}}, align 8

  unsigned long f =  alignof(float);
  // CHECK: store i64 4, ptr {{%.*}}, align 8

  unsigned long d =  alignof(double);
  // CHECK: store i64 8, ptr {{%.*}}, align 8

  unsigned long iArr =  alignof(int[5]);
  // CHECK: store i64 4, ptr {{%.*}}, align 8

  unsigned long dArr =  alignof(double[5]);
  // CHECK: store i64 8, ptr {{%.*}}, align 8
}
