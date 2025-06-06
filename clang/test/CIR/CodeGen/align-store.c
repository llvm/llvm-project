// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// PR5279 - Reduced alignment on typedef.
typedef int myint __attribute__((aligned(1)));

void test1(myint *p) {
  *p = 0;
}

// CIR: cir.func @test1
// CIR:   cir.store align(1)

// LLVM: @test1
// LLVM:  store i32 0, ptr {{.*}}, align 1
// LLVM:  ret void

// OGCG: @test1
// OGCG:  store i32 0, ptr {{.*}}, align 1
// OGCG:  ret void

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef struct
{
   uint16_t i16;
   uint32_t i32;
   uint16_t i16_2;
   uint32_t i32_2;
} StructA;

void test2(StructA* p) {
  p->i16 = 1;
  p->i32 = 2;
  p->i16_2 = 3;
  p->i32_2 = 4;
}

// CIR-LABEL: @test2
// CIR:  cir.store align(4) %{{.*}}, %{{.*}} : !u16i, !cir.ptr<!u16i>
// CIR:  cir.store align(4) %{{.*}}, %{{.*}} : !u32i, !cir.ptr<!u32i>
// CIR:  cir.store align(4) %{{.*}}, %{{.*}} : !u16i, !cir.ptr<!u16i>
// CIR:  cir.store align(4) %{{.*}}, %{{.*}} : !u32i, !cir.ptr<!u32i>

// LLVM: @test2
// LLVM: store i16 1, ptr {{.*}}, align 4
// LLVM: store i32 2, ptr {{.*}}, align 4
// LLVM: store i16 3, ptr {{.*}}, align 4
// LLVM: store i32 4, ptr {{.*}}, align 4

// OGCG: @test2
// OGCG: store i16 1, ptr {{.*}}, align 4
// OGCG: store i32 2, ptr {{.*}}, align 4
// OGCG: store i16 3, ptr {{.*}}, align 4
// OGCG: store i32 4, ptr {{.*}}, align 4

typedef struct {
  short a;
  short b;
  short c;
  short d;
  long e;   // Make the struct 8-byte aligned
} StructB;

void test3(StructB *ptr) {
  ptr->a = 1;  // align 8
  ptr->b = 2;  // align 2
  ptr->c = 3;  // align 4
  ptr->d = 4;  // align 2
}

// CIR-LABEL: @test3
// CIR:  cir.store align(8) %{{.*}}, %{{.*}} : !s16i, !cir.ptr<!s16i>
// CIR:  cir.store align(2) %{{.*}}, %{{.*}} : !s16i, !cir.ptr<!s16i>
// CIR:  cir.store align(4) %{{.*}}, %{{.*}} : !s16i, !cir.ptr<!s16i>
// CIR:  cir.store align(2) %{{.*}}, %{{.*}} : !s16i, !cir.ptr<!s16i>

// LLVM: @test3
// LLVM: store i16 1, ptr {{.*}}, align 8
// LLVM: store i16 2, ptr {{.*}}, align 2
// LLVM: store i16 3, ptr {{.*}}, align 4
// LLVM: store i16 4, ptr {{.*}}, align 2

// OGCG: @test3
// OGCG: store i16 1, ptr {{.*}}, align 8
// OGCG: store i16 2, ptr {{.*}}, align 2
// OGCG: store i16 3, ptr {{.*}}, align 4
// OGCG: store i16 4, ptr {{.*}}, align 2
