// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

union U1 {
  int n;
  char c;
};

// CIR:  !rec_U1 = !cir.record<union "U1" {!s32i, !s8i}>
// LLVM: %union.U1 = type { i32 }
// OGCG: %union.U1 = type { i32 }

union U2 {
  char b;
  short s;
  int i;
  float f;
  double d;
};

// CIR:  !rec_U2 = !cir.record<union "U2" {!s8i, !s16i, !s32i, !cir.float, !cir.double}>
// LLVM: %union.U2 = type { double }
// OGCG: %union.U2 = type { double }

union IncompleteU *p;

// CIR:  cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteU>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

void f1(void) {
  union IncompleteU *p;
}

// CIR:      cir.func @f1()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteU>, !cir.ptr<!cir.ptr<!rec_IncompleteU>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @f1()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f1()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

int f2(void) {
  union U1 u;
  u.n = 42;
  return u.n;
}

// CIR:      cir.func @f2() -> !s32i
// CIR-NEXT:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %1 = cir.alloca !rec_U1, !cir.ptr<!rec_U1>, ["u"] {alignment = 4 : i64}
// CIR-NEXT:   %2 = cir.const #cir.int<42> : !s32i
// CIR-NEXT:   %3 = cir.get_member %1[0] {name = "n"} : !cir.ptr<!rec_U1> -> !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %2, %3 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %4 = cir.get_member %1[0] {name = "n"} : !cir.ptr<!rec_U1> -> !cir.ptr<!s32i>
// CIR-NEXT:   %5 = cir.load %4 : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.store %5, %0 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %6 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %6 : !s32i

// LLVM:      define i32 @f2()
// LLVM-NEXT:   %1 = alloca i32, i64 1, align 4
// LLVM-NEXT:   %2 = alloca %union.U1, i64 1, align 4
// LLVM-NEXT:   store i32 42, ptr %2, align 4
// LLVM-NEXT:   %3 = load i32, ptr %2, align 4
// LLVM-NEXT:   store i32 %3, ptr %1, align 4
// LLVM-NEXT:   %4 = load i32, ptr %1, align 4
// LLVM-NEXT:   ret i32 %4

//      OGCG: define dso_local i32 @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT: %u = alloca %union.U1, align 4
// OGCG-NEXT: store i32 42, ptr %u, align 4
// OGCG-NEXT: %0 = load i32, ptr %u, align 4
// OGCG-NEXT: ret i32 %0


void shouldGenerateUnionAccess(union U2 u) {
  u.b = 0;
  u.b;
  u.i = 1;
  u.i;
  u.f = 0.1F;
  u.f;
  u.d = 0.1;
  u.d;
}

// CIR:      cir.func @shouldGenerateUnionAccess(%arg0: !rec_U2
// CIR-NEXT:   %0 = cir.alloca !rec_U2, !cir.ptr<!rec_U2>, ["u", init] {alignment = 8 : i64}
// CIR-NEXT:   cir.store %arg0, %0 : !rec_U2, !cir.ptr<!rec_U2>
// CIR-NEXT:   %1 = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   %2 = cir.cast(integral, %1 : !s32i), !s8i
// CIR-NEXT:   %3 = cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s8i>
// CIR-NEXT:   cir.store %2, %3 : !s8i, !cir.ptr<!s8i>
// CIR-NEXT:   %4 = cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s8i>
// CIR-NEXT:   %5 = cir.load %4 : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %6 = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   %7 = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %6, %7 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %8 = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U2> -> !cir.ptr<!s32i>
// CIR-NEXT:   %9 = cir.load %8 : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   %10 = cir.const #cir.fp<1.000000e-01> : !cir.float
// CIR-NEXT:   %11 = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.float>
// CIR-NEXT:   cir.store %10, %11 : !cir.float, !cir.ptr<!cir.float>
// CIR-NEXT:   %12 = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.float>
// CIR-NEXT:   %13 = cir.load %12 : !cir.ptr<!cir.float>, !cir.float
// CIR-NEXT:   %14 = cir.const #cir.fp<1.000000e-01> : !cir.double
// CIR-NEXT:   %15 = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.double>
// CIR-NEXT:   cir.store %14, %15 : !cir.double, !cir.ptr<!cir.double>
// CIR-NEXT:   %16 = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U2> -> !cir.ptr<!cir.double>
// CIR-NEXT:   %17 = cir.load %16 : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:   cir.return

// LLVM:      define void @shouldGenerateUnionAccess(%union.U2 %0) {
// LLVM-NEXT:   %2 = alloca %union.U2, i64 1, align 8
// LLVM-NEXT:   store %union.U2 %0, ptr %2, align 8
// LLVM-NEXT:   store i8 0, ptr %2, align 1
// LLVM-NEXT:   %3 = load i8, ptr %2, align 1
// LLVM-NEXT:   store i32 1, ptr %2, align 4
// LLVM-NEXT:   %4 = load i32, ptr %2, align 4
// LLVM-NEXT:   store float 0x3FB99999A0000000, ptr %2, align 4
// LLVM-NEXT:   %5 = load float, ptr %2, align 4
// LLVM-NEXT:   store double 1.000000e-01, ptr %2, align 8
// LLVM-NEXT:   %6 = load double, ptr %2, align 8
// LLVM-NEXT:   ret void

// OGCG:      define dso_local void @shouldGenerateUnionAccess(i64 %u.coerce) #0 {
// OGCG-NEXT: entry:
// OGCG-NEXT:   %u = alloca %union.U2, align 8
// OGCG-NEXT:   %coerce.dive = getelementptr inbounds nuw %union.U2, ptr %u, i32 0, i32 0
// OGCG-NEXT:   store i64 %u.coerce, ptr %coerce.dive, align 8
// OGCG-NEXT:   store i8 0, ptr %u, align 8
// OGCG-NEXT:   %0 = load i8, ptr %u, align 8
// OGCG-NEXT:   store i32 1, ptr %u, align 8
// OGCG-NEXT:   %1 = load i32, ptr %u, align 8
// OGCG-NEXT:   store float 0x3FB99999A0000000, ptr %u, align 8
// OGCG-NEXT:   %2 = load float, ptr %u, align 8
// OGCG-NEXT:   store double 1.000000e-01, ptr %u, align 8
// OGCG-NEXT:   %3 = load double, ptr %u, align 8
// OGCG-NEXT:   ret void