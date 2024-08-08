// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare -o %t.cir %s 2>&1 | FileCheck --check-prefixes=CIR-BEFORE,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare -o %t.cir %s 2>&1 | FileCheck --check-prefixes=CIR-AFTER,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=LLVM,CHECK %s

#include <stdbool.h>

volatile double _Complex cd;
volatile float _Complex cf;
volatile int _Complex ci;
volatile short _Complex cs;
volatile double sd;
volatile int si;
volatile bool b;

void scalar_to_complex() {
  cd = sd;
  ci = si;
  cd = si;
  ci = sd;
}

// CHECK-LABEL: @scalar_to_complex()

// CIR-BEFORE: %{{.+}} = cir.cast(float_to_complex, %{{.+}} : !cir.double), !cir.complex<!cir.double>

//      CIR-AFTER: %[[#REAL:]] = cir.load volatile %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

// CIR-BEFORE: %{{.+}} = cir.cast(int_to_complex, %{{.+}} : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#REAL:]] = cir.load volatile %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      CIR-BEFORE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.double
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(float_to_complex, %[[#A]] : !cir.double), !cir.complex<!cir.double>

//      CIR-AFTER: %[[#A:]] = cir.load volatile %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-AFTER-NEXT: %[[#REAL:]] = cir.cast(int_to_float, %[[#A]] : !s32i), !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      CIR-BEFORE: %[[#A:]] = cir.cast(float_to_int, %{{.+}} : !cir.double), !s32i
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(int_to_complex, %[[#A]] : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#A:]] = cir.load volatile %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CIR-AFTER-NEXT: %[[#REAL:]] = cir.cast(float_to_int, %[[#A]] : !cir.double), !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#REAL:]] = load volatile double, ptr @sd, align 8
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } undef, double %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double 0.000000e+00, 1

//      LLVM: %[[#REAL:]] = load volatile i32, ptr @si, align 4
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } undef, i32 %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 0, 1

//      LLVM: %[[#A:]] = load volatile i32, ptr @si, align 4
// LLVM-NEXT: %[[#REAL:]] = sitofp i32 %[[#A]] to double
// LLVM-NEXT: %[[#B:]] = insertvalue { double, double } undef, double %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#B]], double 0.000000e+00, 1

//      LLVM: %[[#A:]] = load volatile double, ptr @sd, align 8
// LLVM-NEXT: %[[#REAL:]] = fptosi double %[[#A]] to i32
// LLVM-NEXT: %[[#B:]] = insertvalue { i32, i32 } undef, i32 %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#B]], i32 0, 1

// CHECK: }

void complex_to_scalar() {
  sd = (double)cd;
  si = (int)ci;
  sd = (double)ci;
  si = (int)cd;
}

// CHECK-LABEL: @complex_to_scalar()

// CIR-BEFORE: %{{.+}} = cir.cast(float_complex_to_real, %{{.+}} : !cir.complex<!cir.double>), !cir.double

// CIR-AFTER: %{{.+}} = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double

// LLVM: %{{.+}} = extractvalue { double, double } %{{.+}}, 0

// CIR-BEFORE: %{{.+}} = cir.cast(int_complex_to_real, %{{.+}} : !cir.complex<!s32i>), !s32i

// CIR-AFTER: %{{.+}} = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i

// LLVM: %{{.+}} = extractvalue { i32, i32 } %{{.+}}, 0

//      CIR-BEFORE: %[[#A:]] = cir.cast(int_complex_to_real, %{{.+}} : !cir.complex<!s32i>), !s32i
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(int_to_float, %[[#A]] : !s32i), !cir.double

//      CIR-AFTER: %[[#A:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.cast(int_to_float, %[[#A]] : !s32i), !cir.double

//      LLVM: %[[#A:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT: %{{.+}} = sitofp i32 %[[#A]] to double

//      CIR-BEFORE: %[[#A:]] = cir.cast(float_complex_to_real, %{{.+}} : !cir.complex<!cir.double>), !cir.double
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(float_to_int, %[[#A]] : !cir.double), !s32i

//      CIR-AFTER: %[[#A:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.cast(float_to_int, %[[#A]] : !cir.double), !s32i

//      LLVM: %[[#A:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT: %{{.+}} = fptosi double %[[#A]] to i32

// CHECK: }

void complex_to_bool() {
  b = (bool)cd;
  b = (bool)ci;
}

// CHECK-LABEL: @complex_to_bool()

// CIR-BEFORE: %{{.+}} = cir.cast(float_complex_to_bool, %{{.+}} : !cir.complex<!cir.double>), !cir.bool

//      CIR-AFTER: %[[#REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!cir.double> -> !cir.double
// CIR-AFTER-NEXT: %[[#RB:]] = cir.cast(float_to_bool, %[[#REAL]] : !cir.double), !cir.bool
// CIR-AFTER-NEXT: %[[#IB:]] = cir.cast(float_to_bool, %[[#IMAG]] : !cir.double), !cir.bool
// CIR-AFTER-NEXT: %{{.+}} = cir.ternary(%[[#RB]], true {
// CIR-AFTER-NEXT:   %[[#A:]] = cir.const #true
// CIR-AFTER-NEXT:   cir.yield %[[#A]] : !cir.bool
// CIR-AFTER-NEXT: }, false {
// CIR-AFTER-NEXT:   %[[#B:]] = cir.ternary(%[[#IB]], true {
// CIR-AFTER-NEXT:     %[[#C:]] = cir.const #true
// CIR-AFTER-NEXT:     cir.yield %[[#C]] : !cir.bool
// CIR-AFTER-NEXT:   }, false {
// CIR-AFTER-NEXT:     %[[#D:]] = cir.const #false
// CIR-AFTER-NEXT:     cir.yield %[[#D]] : !cir.bool
// CIR-AFTER-NEXT:   }) : (!cir.bool) -> !cir.bool
// CIR-AFTER-NEXT:   cir.yield %[[#B]] : !cir.bool
// CIR-AFTER-NEXT: }) : (!cir.bool) -> !cir.bool

//      LLVM:   %[[#REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT:   %[[#IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT:   %[[#RB:]] = fcmp une double %[[#REAL]], 0.000000e+00
// LLVM-NEXT:   %[[#IB:]] = fcmp une double %[[#IMAG]], 0.000000e+00
// LLVM-NEXT:   br i1 %[[#RB]], label %[[#LABEL_RB:]], label %[[#LABEL_RB_NOT:]]
//      LLVM: [[#LABEL_RB]]:
// LLVM-NEXT:   br label %[[#LABEL_EXIT:]]
//      LLVM: [[#LABEL_RB_NOT]]:
// LLVM-NEXT:   br i1 %[[#IB]], label %[[#LABEL_IB:]], label %[[#LABEL_IB_NOT:]]
//      LLVM: [[#LABEL_IB]]:
// LLVM-NEXT:   br label %[[#LABEL_A:]]
//      LLVM: [[#LABEL_IB_NOT]]:
// LLVM-NEXT:   br label %[[#LABEL_A]]
//      LLVM: [[#LABEL_A]]:
// LLVM-NEXT:   %[[#A:]] = phi i8 [ 0, %[[#LABEL_IB_NOT]] ], [ 1, %[[#LABEL_IB]] ]
// LLVM-NEXT:   br label %[[#LABEL_B:]]
//      LLVM: [[#LABEL_B]]:
// LLVM-NEXT:   br label %[[#LABEL_EXIT]]
//      LLVM: [[#LABEL_EXIT]]:
// LLVM-NEXT:   %{{.+}} = phi i8 [ %[[#A]], %[[#LABEL_B]] ], [ 1, %[[#LABEL_RB]] ]
// LLVM-NEXT:   br label %{{.+}}

// CIR-BEFORE: %{{.+}} = cir.cast(int_complex_to_bool, %{{.+}} : !cir.complex<!s32i>), !cir.bool

//      CIR-AFTER: %[[#REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-NEXT: %[[#RB:]] = cir.cast(int_to_bool, %[[#REAL]] : !s32i), !cir.bool
// CIR-AFTER-NEXT: %[[#IB:]] = cir.cast(int_to_bool, %[[#IMAG]] : !s32i), !cir.bool
// CIR-AFTER-NEXT: %{{.+}} = cir.ternary(%[[#RB]], true {
// CIR-AFTER-NEXT:   %[[#A:]] = cir.const #true
// CIR-AFTER-NEXT:   cir.yield %[[#A]] : !cir.bool
// CIR-AFTER-NEXT: }, false {
// CIR-AFTER-NEXT:   %[[#B:]] = cir.ternary(%[[#IB]], true {
// CIR-AFTER-NEXT:     %[[#C:]] = cir.const #true
// CIR-AFTER-NEXT:     cir.yield %[[#C]] : !cir.bool
// CIR-AFTER-NEXT:   }, false {
// CIR-AFTER-NEXT:     %[[#D:]] = cir.const #false
// CIR-AFTER-NEXT:     cir.yield %[[#D]] : !cir.bool
// CIR-AFTER-NEXT:   }) : (!cir.bool) -> !cir.bool
// CIR-AFTER-NEXT:   cir.yield %[[#B]] : !cir.bool
// CIR-AFTER-NEXT: }) : (!cir.bool) -> !cir.bool

//      LLVM:   %[[#REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT:   %[[#IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT:   %[[#RB:]] = icmp ne i32 %[[#REAL]], 0
// LLVM-NEXT:   %[[#IB:]] = icmp ne i32 %[[#IMAG]], 0
// LLVM-NEXT:   br i1 %[[#RB]], label %[[#LABEL_RB:]], label %[[#LABEL_RB_NOT:]]
//      LLVM: [[#LABEL_RB]]:
// LLVM-NEXT:   br label %[[#LABEL_EXIT:]]
//      LLVM: [[#LABEL_RB_NOT]]:
// LLVM-NEXT:   br i1 %[[#IB]], label %[[#LABEL_IB:]], label %[[#LABEL_IB_NOT:]]
//      LLVM: [[#LABEL_IB]]:
// LLVM-NEXT:   br label %[[#LABEL_A:]]
//      LLVM: [[#LABEL_IB_NOT]]:
// LLVM-NEXT:   br label %[[#LABEL_A]]
//      LLVM: [[#LABEL_A]]:
// LLVM-NEXT:   %[[#A:]] = phi i8 [ 0, %[[#LABEL_IB_NOT]] ], [ 1, %[[#LABEL_IB]] ]
// LLVM-NEXT:   br label %[[#LABEL_B:]]
//      LLVM: [[#LABEL_B]]:
// LLVM-NEXT:   br label %[[#LABEL_EXIT]]
//      LLVM: [[#LABEL_EXIT]]:
// LLVM-NEXT:   %{{.+}} = phi i8 [ %[[#A]], %[[#LABEL_B]] ], [ 1, %[[#LABEL_RB]] ]
// LLVM-NEXT:   br label %{{.+}}

// CHECK: }
