// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Struct {
  int a;
  float b;
  double c;
  bool d;
};

void offset_of_builtin() {
  unsigned long a = __builtin_offsetof(Struct, a);
  unsigned long b = __builtin_offsetof(Struct, b);
  unsigned long c = __builtin_offsetof(Struct, c);
  unsigned long d = __builtin_offsetof(Struct, d);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["c", init]
// CIR: %[[D_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["d", init]
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !u64i
// CIR: cir.store {{.*}} %[[CONST_0]], %[[A_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_4:.*]] = cir.const #cir.int<4> : !u64i
// CIR: cir.store {{.*}} %[[CONST_4]], %[[B_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_8:.*]] = cir.const #cir.int<8> : !u64i
// CIR: cir.store {{.*}} %[[CONST_8]], %[[C_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_16:.*]] = cir.const #cir.int<16> : !u64i
// CIR: cir.store {{.*}} %[[CONST_16]], %[[D_ADDR]] : !u64i, !cir.ptr<!u64i>

// LLVM: %[[A_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[D_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: store i64 0, ptr %[[A_ADDR]], align 8
// LLVM: store i64 4, ptr %[[B_ADDR]], align 8
// LLVM: store i64 8, ptr %[[C_ADDR]], align 8
// LLVM: store i64 16, ptr %[[D_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[B_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[C_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[D_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 0, ptr %[[A_ADDR]], align 8
// OGCG: store i64 4, ptr %[[B_ADDR]], align 8
// OGCG: store i64 8, ptr %[[C_ADDR]], align 8
// OGCG: store i64 16, ptr %[[D_ADDR]], align 8

struct StructWithArray {
  Struct array[4][4];
};

void offset_of_builtin_from_array_element() {
  unsigned long a = __builtin_offsetof(StructWithArray, array[0][0].a);
  unsigned long b = __builtin_offsetof(StructWithArray, array[1][1].b);
  unsigned long c = __builtin_offsetof(StructWithArray, array[2][2].c);
  unsigned long d = __builtin_offsetof(StructWithArray, array[3][3].d);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["c", init]
// CIR: %[[D_ADDR:.*]] = cir.alloca !u64i, !cir.ptr<!u64i>, ["d", init]
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !u64i
// CIR: cir.store {{.*}} %[[CONST_0]], %[[A_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_124:.*]] = cir.const #cir.int<124> : !u64i
// CIR: cir.store {{.*}} %[[CONST_124]], %[[B_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_248:.*]] = cir.const #cir.int<248> : !u64i
// CIR: cir.store {{.*}} %[[CONST_248]], %[[C_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[CONST_376:.*]] = cir.const #cir.int<376> : !u64i
// CIR: cir.store {{.*}} %[[CONST_376]], %[[D_ADDR]] : !u64i, !cir.ptr<!u64i>

// LLVM: %[[A_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: %[[D_ADDR:.*]] = alloca i64, i64 1, align 8
// LLVM: store i64 0, ptr %[[A_ADDR]], align 8
// LLVM: store i64 124, ptr %[[B_ADDR]], align 8
// LLVM: store i64 248, ptr %[[C_ADDR]], align 8
// LLVM: store i64 376, ptr %[[D_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[B_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[C_ADDR:.*]] = alloca i64, align 8
// OGCG: %[[D_ADDR:.*]] = alloca i64, align 8
// OGCG: store i64 0, ptr %[[A_ADDR]], align 8
// OGCG: store i64 124, ptr %[[B_ADDR]], align 8
// OGCG: store i64 248, ptr %[[C_ADDR]], align 8
// OGCG: store i64 376, ptr %[[D_ADDR]], align 8
