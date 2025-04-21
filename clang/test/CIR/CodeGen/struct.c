// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// For LLVM IR checks, the structs are defined before the variables, so these
// checks are at the top.
// LLVM: %struct.CompleteS = type { i32, i8 }
// LLVM: %struct.OuterS = type { %struct.InnerS, i32 }
// LLVM: %struct.InnerS = type { i32, i8 }
// LLVM: %struct.PackedS = type <{ i32, i8 }>
// LLVM: %struct.PackedAndPaddedS = type <{ i32, i8, i8 }>
// OGCG: %struct.CompleteS = type { i32, i8 }
// OGCG: %struct.OuterS = type { %struct.InnerS, i32 }
// OGCG: %struct.InnerS = type { i32, i8 }
// OGCG: %struct.PackedS = type <{ i32, i8 }>
// OGCG: %struct.PackedAndPaddedS = type <{ i32, i8, i8 }>

struct IncompleteS *p;

// CIR:      cir.global external @p = #cir.ptr<null> : !cir.ptr<!cir.record<struct
// CIR-SAME:     "IncompleteS" incomplete>>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

struct CompleteS {
  int a;
  char b;
} cs;

// CIR:       cir.global external @cs = #cir.zero : !cir.record<struct
// CIR-SAME:      "CompleteS" {!s32i, !s8i}>
// LLVM:      @cs = dso_local global %struct.CompleteS zeroinitializer
// OGCG:      @cs = global %struct.CompleteS zeroinitializer, align 4

struct InnerS {
  int a;
  char b;
};

struct OuterS {
  struct InnerS is;
  int c;
};

struct OuterS os;

// CIR:       cir.global external @os = #cir.zero : !cir.record<struct
// CIR-SAME:      "OuterS" {!cir.record<struct "InnerS" {!s32i, !s8i}>, !s32i}>
// LLVM:      @os = dso_local global %struct.OuterS zeroinitializer
// OGCG:      @os = global %struct.OuterS zeroinitializer, align 4

#pragma pack(push)
#pragma pack(1)

struct PackedS {
  int  a0;
  char a1;
} ps;

// CIR:       cir.global external @ps = #cir.zero : !cir.record<struct "PackedS"
// CIR-SAME:      packed {!s32i, !s8i}>
// LLVM:      @ps = dso_local global %struct.PackedS zeroinitializer
// OGCG:      @ps = global %struct.PackedS zeroinitializer, align 1

struct PackedAndPaddedS {
  int  b0;
  char b1;
} __attribute__((aligned(2))) pps;

// CIR:       cir.global external @pps = #cir.zero : !cir.record<struct
// CIR-SAME:      "PackedAndPaddedS" packed padded {!s32i, !s8i, !u8i}>
// LLVM:      @pps = dso_local global %struct.PackedAndPaddedS zeroinitializer
// OGCG:      @pps = global %struct.PackedAndPaddedS zeroinitializer, align 2

#pragma pack(pop)

void f(void) {
  struct IncompleteS *p;
}

// CIR:      cir.func @f()
// CIR-NEXT:   cir.alloca !cir.ptr<!cir.record<struct "IncompleteS" incomplete>>,
// CIR-SAME:       !cir.ptr<!cir.ptr<!cir.record<struct
// CIR-SAME:       "IncompleteS" incomplete>>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @f()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

void f2(void) {
  struct CompleteS s;
}

// CIR:      cir.func @f2()
// CIR-NEXT:   cir.alloca !cir.record<struct "CompleteS" {!s32i, !s8i}>,
// CIR-SAME:       !cir.ptr<!cir.record<struct "CompleteS" {!s32i, !s8i}>>,
// CIR-SAME:       ["s"] {alignment = 4 : i64}
// CIR-NEXT:   cir.return

// LLVM:      define void @f2()
// LLVM-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, align 4
// OGCG-NEXT:   ret void
