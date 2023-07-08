// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int foo(int i);

int foo(int i) {
  i;
  return i;
}

//      CIR: module @"{{.*}}basic.c" attributes {{{.*}}cir.lang = #cir.lang<c>
// CIR-NEXT: cir.func @foo(%arg0: !s32i loc({{.*}})) -> !s32i
// CIR-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT: %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT: cir.store %arg0, %0 : !s32i, cir.ptr <!s32i>
// CIR-NEXT: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: %3 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: cir.store %3, %1 : !s32i, cir.ptr <!s32i>
// CIR-NEXT: %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: cir.return %4 : !s32i

int f2(void) { return 3; }

// CIR: cir.func @f2() -> !s32i
// CIR-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT: %1 = cir.const(#cir.int<3> : !s32i) : !s32i
// CIR-NEXT: cir.store %1, %0 : !s32i, cir.ptr <!s32i>
// CIR-NEXT: %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: cir.return %2 : !s32i

// LLVM: define i32 @f2()
// LLVM-NEXT:  %1 = alloca i32, i64 1, align 4
// LLVM-NEXT:  store i32 3, ptr %1, align 4
// LLVM-NEXT:  %2 = load i32, ptr %1, align 4
// LLVM-NEXT:  ret i32 %2



int f3(void) {
  int i = 3;
  return i;
}

// CIR: cir.func @f3() -> !s32i
// CIR-NEXT: %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT: %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT: %2 = cir.const(#cir.int<3> : !s32i) : !s32i
// CIR-NEXT: cir.store %2, %1 : !s32i, cir.ptr <!s32i>
// CIR-NEXT: %3 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: cir.store %3, %0 : !s32i, cir.ptr <!s32i>
// CIR-NEXT: %4 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CIR-NEXT: cir.return %4 : !s32i
