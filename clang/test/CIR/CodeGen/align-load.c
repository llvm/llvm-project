// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  char b;
  short s;
  int i;
  float f;
  double d;
};

void accessStruct(struct S u) {
  u.b;
  u.s;
  u.i;
  u.f;
  u.d;
}

// CIR: cir.func @accessStruct
// CIR:   cir.load align(8)
// CIR:   cir.load align(2)
// CIR:   cir.load align(4)
// CIR:   cir.load align(8)
// CIR:   cir.load align(8)

// LLVM: define{{.*}} @accessStruct
// LLVM:   load i8, ptr {{.*}}, align 8
// LLVM:   load i16, ptr {{.*}}, align 2
// LLVM:   load i32, ptr {{.*}}, align 4
// LLVM:   load float, ptr {{.*}}, align 8
// LLVM:   load double, ptr {{.*}}, align 8

// OGCG: define{{.*}} @accessStruct
// OGCG:   load i8, ptr {{.*}}, align 8
// OGCG:   load i16, ptr {{.*}}, align 2
// OGCG:   load i32, ptr {{.*}}, align 4
// OGCG:   load float, ptr {{.*}}, align 8
// OGCG:   load double, ptr {{.*}}, align 8

union U {
  char b;
  short s;
  int i;
  float f;
  double d;
};

void accessUnion(union U u) {
  u.b;
  u.s;
  u.i;
  u.f;
  u.d;
}

// CIR: cir.func @accessUnion
// CIR:   cir.load align(8)
// CIR:   cir.load align(8)
// CIR:   cir.load align(8)
// CIR:   cir.load align(8)
// CIR:   cir.load align(8)

// LLVM: define{{.*}} @accessUnion
// LLVM:   load i8, ptr {{.*}}, align 8
// LLVM:   load i16, ptr {{.*}}, align 8
// LLVM:   load i32, ptr {{.*}}, align 8
// LLVM:   load float, ptr {{.*}}, align 8
// LLVM:   load double, ptr {{.*}}, align 8

// OGCG: define{{.*}} @accessUnion
// OGCG:   load i8, ptr {{.*}}, align 8
// OGCG:   load i16, ptr {{.*}}, align 8
// OGCG:   load i32, ptr {{.*}}, align 8
// OGCG:   load float, ptr {{.*}}, align 8
// OGCG:   load double, ptr {{.*}}, align 8

// PR5279 - Reduced alignment on typedef.
typedef int myint __attribute__((aligned(1)));

int loadAligned(myint *p) {
  return *p;
}

// CIR: cir.func @loadAligned
// CIR:   cir.load align(1)

// LLVM: @loadAligned
// LLVM:   load i32, ptr {{.*}}, align 1

// OGCG: @loadAligned
// OGCG:   load i32, ptr {{.*}}, align 1
