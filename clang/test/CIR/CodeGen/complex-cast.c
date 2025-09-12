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

//      CIR-AFTER: %[[#REAL:]] = cir.load volatile{{.*}}  %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

// CIR-BEFORE: %{{.+}} = cir.cast(int_to_complex, %{{.+}} : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#REAL:]] = cir.load volatile{{.*}}  %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      CIR-BEFORE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.double
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(float_to_complex, %[[#A]] : !cir.double), !cir.complex<!cir.double>

//      CIR-AFTER: %[[#A:]] = cir.load volatile{{.*}}  %{{.+}} : !cir.ptr<!s32i>, !s32i
// CIR-AFTER-NEXT: %[[#REAL:]] = cir.cast(int_to_float, %[[#A]] : !s32i), !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      CIR-BEFORE: %[[#A:]] = cir.cast(float_to_int, %{{.+}} : !cir.double), !s32i
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(int_to_complex, %[[#A]] : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#A:]] = cir.load volatile{{.*}}  %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CIR-AFTER-NEXT: %[[#REAL:]] = cir.cast(float_to_int, %[[#A]] : !cir.double), !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#REAL:]] = load volatile double, ptr @sd, align 8
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } {{.*}}, double %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double 0.000000e+00, 1

//      LLVM: %[[#REAL:]] = load volatile i32, ptr @si, align 4
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } {{.*}}, i32 %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 0, 1

//      LLVM: %[[#A:]] = load volatile i32, ptr @si, align 4
// LLVM-NEXT: %[[#REAL:]] = sitofp i32 %[[#A]] to double
// LLVM-NEXT: %[[#B:]] = insertvalue { double, double } {{.*}}, double %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#B]], double 0.000000e+00, 1

//      LLVM: %[[#A:]] = load volatile double, ptr @sd, align 8
// LLVM-NEXT: %[[#REAL:]] = fptosi double %[[#A]] to i32
// LLVM-NEXT: %[[#B:]] = insertvalue { i32, i32 } {{.*}}, i32 %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#B]], i32 0, 1

// CHECK: }

void scalar_to_complex_explicit() {
  cd = (double _Complex)sd;
  ci = (int _Complex)si;
  cd = (double _Complex)si;
  ci = (int _Complex)sd;
}

// CHECK-LABEL: @scalar_to_complex_explicit()

// CIR-BEFORE: %{{.+}} = cir.cast(float_to_complex, %{{.+}} : !cir.double), !cir.complex<!cir.double>

//      CIR-AFTER: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %{{.+}}, %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      LLVM: %[[#A:]] = insertvalue { double, double } {{.*}}, double %{{.+}}, 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double 0.000000e+00, 1

// CIR-BEFORE: %{{.+}} = cir.cast(int_to_complex, %{{.+}} : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %{{.+}}, %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#A:]] = insertvalue { i32, i32 } {{.*}}, i32 %{{.+}}, 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 0, 1

//      CIR-BEFORE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.double
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(float_to_complex, %[[#A]] : !cir.double), !cir.complex<!cir.double>

//      CIR-AFTER: %[[#REAL:]] = cir.cast(int_to_float, %11 : !s32i), !cir.double
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>

//      LLVM: %[[#REAL:]] = sitofp i32 %{{.+}} to double
// LLVM-NEXT: %[[#A:]] = insertvalue { double, double } {{.*}}, double %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { double, double } %[[#A]], double 0.000000e+00, 1

//      CIR-BEFORE: %[[#A:]] = cir.cast(float_to_int, %{{.+}} : !cir.double), !s32i
// CIR-BEFORE-NEXT: %{{.+}} = cir.cast(int_to_complex, %[[#A]] : !s32i), !cir.complex<!s32i>

//      CIR-AFTER: %[[#REAL:]] = cir.cast(float_to_int, %{{.+}} : !cir.double), !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.const #cir.int<0> : !s32i
// CIR-AFTER-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>

//      LLVM: %[[#REAL:]] = fptosi double %{{.+}} to i32
// LLVM-NEXT: %[[#A:]] = insertvalue { i32, i32 } {{.*}}, i32 %[[#REAL]], 0
// LLVM-NEXT: %{{.+}} = insertvalue { i32, i32 } %[[#A]], i32 0, 1

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
// CIR-AFTER-NEXT: %[[#A:]] = cir.const #true
// CIR-AFTER-NEXT: %{{.+}} = cir.select if %[[#RB]] then %[[#A]] else %[[#IB]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool

//      LLVM:   %[[#REAL:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT:   %[[#IMAG:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT:   %[[#RB:]] = fcmp une double %[[#REAL]], 0.000000e+00
// LLVM-NEXT:   %[[#IB:]] = fcmp une double %[[#IMAG]], 0.000000e+00
// LLVM-NEXT:   %{{.+}} = or i1 %[[#RB]], %[[#IB]]

// CIR-BEFORE: %{{.+}} = cir.cast(int_complex_to_bool, %{{.+}} : !cir.complex<!s32i>), !cir.bool

//      CIR-AFTER: %[[#REAL:]] = cir.complex.real %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-NEXT: %[[#IMAG:]] = cir.complex.imag %{{.+}} : !cir.complex<!s32i> -> !s32i
// CIR-AFTER-NEXT: %[[#RB:]] = cir.cast(int_to_bool, %[[#REAL]] : !s32i), !cir.bool
// CIR-AFTER-NEXT: %[[#IB:]] = cir.cast(int_to_bool, %[[#IMAG]] : !s32i), !cir.bool
// CIR-AFTER-NEXT: %[[#A:]] = cir.const #true
// CIR-AFTER-NEXT: %{{.+}} = cir.select if %[[#RB]] then %[[#A]] else %[[#IB]] : (!cir.bool, !cir.bool, !cir.bool) -> !cir.bool

//      LLVM:   %[[#REAL:]] = extractvalue { i32, i32 } %{{.+}}, 0
// LLVM-NEXT:   %[[#IMAG:]] = extractvalue { i32, i32 } %{{.+}}, 1
// LLVM-NEXT:   %[[#RB:]] = icmp ne i32 %[[#REAL]], 0
// LLVM-NEXT:   %[[#IB:]] = icmp ne i32 %[[#IMAG]], 0
// LLVM-NEXT:   %{{.+}} = or i1 %[[#RB]], %[[#IB]]

// CHECK: }

struct CX {
  double real;
  double imag;
};

void lvalue_to_rvalue_bitcast() {
   struct CX a;
   double _Complex b = __builtin_bit_cast(double _Complex, a);
}

// CHECK-LABEL: @lvalue_to_rvalue_bitcast()

// CIR-BEFORE: %{{.+}} = cir.cast(bitcast, %{{.+}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// CIR-AFTER: %{{.+}} = cir.cast(bitcast, %{{.+}} : !cir.ptr<!rec_CX>), !cir.ptr<!cir.complex<!cir.double>>

// LLVM: %[[PTR_ADDR:.*]] = alloca %struct.CX, i64 1, align 8
// LLVM: %[[COMPLEX_ADDR:.*]] = alloca { double, double }, i64 1, align 8
// LLVM: %[[PTR_TO_COMPLEX:.*]] = load { double, double }, ptr %[[PTR_ADDR]], align 8
// LLVM: store { double, double } %[[PTR_TO_COMPLEX]], ptr %[[COMPLEX_ADDR]], align 8

// CHECK: }

void complex_to_complex_cast() {
  cd = cf;
  ci = cs;
}

// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-BEFORE: %[[FP_COMPLEX:.*]] = cir.cast(float_complex, %[[TMP]] : !cir.complex<!cir.float>), !cir.complex<!cir.double>

// CIR-AFTER: %[[#REAL:]] = cir.complex.real %{{.*}} : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[#IMAG:]] = cir.complex.imag %{{.*}} : !cir.complex<!cir.float> -> !cir.float
// CIR-AFTER: %[[#REAL_FP_CAST:]] = cir.cast(floating, %[[#REAL]] : !cir.float), !cir.double
// CIR-AFTER: %[[#IMAG_FP_CAST:]] = cir.cast(floating, %[[#IMAG]] : !cir.float), !cir.double
// CIR-AFTER: %{{.*}} = cir.complex.create %[[#REAL_FP_CAST]], %[[#IMAG_FP_CAST]] : !cir.double -> !cir.complex<!cir.double>

// LLVM: %[[#REAL:]] = extractvalue { float, float } %{{.*}}, 0
// LLVM: %[[#IMAG:]] = extractvalue { float, float } %{{.*}}, 1
// LLVM: %[[#REAL_FP_CAST:]] = fpext float %[[#REAL]] to double
// LLVM: %[[#IMAG_FP_CAST:]] = fpext float %[[#IMAG]] to double
// LLVM: %[[TMP:.*]] = insertvalue { double, double } {{.*}}, double %[[#REAL_FP_CAST]], 0
// LLVM: %{{.*}} = insertvalue { double, double } %[[TMP]], double %[[#IMAG_FP_CAST]], 1

// CIR-BEFORE: %[[TMP:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.complex<!s16i>>, !cir.complex<!s16i>
// CIR-BEFORE: %[[INT_COMPLEX:.*]] = cir.cast(int_complex, %[[TMP]] : !cir.complex<!s16i>), !cir.complex<!s32i>

// CIR-AFTER: %[[#REAL:]] = cir.complex.real %{{.*}} : !cir.complex<!s16i> -> !s16i
// CIR-AFTER: %[[#IMAG:]] = cir.complex.imag %{{.*}} : !cir.complex<!s16i> -> !s16i
// CIR-AFTER: %[[#REAL_INT_CAST:]] = cir.cast(integral, %[[#REAL]] : !s16i), !s32i
// CIR-AFTER: %[[#IMAG_INT_CAST:]] = cir.cast(integral, %[[#IMAG]] : !s16i), !s32i
// CIR-AFTER: %{{.*}} = cir.complex.create %[[#REAL_INT_CAST]], %[[#IMAG_INT_CAST]] : !s32i -> !cir.complex<!s32i>

// LLVM: %[[#REAL:]] = extractvalue { i16, i16 } %{{.*}}, 0
// LLVM: %[[#IMAG:]] = extractvalue { i16, i16 } %{{.*}}, 1
// LLVM: %[[#REAL_INT_CAST:]] = sext i16 %[[#REAL]] to i32
// LLVM: %[[#IMAG_INT_CAST:]] = sext i16 %[[#IMAG]] to i32
// LLVM: %[[TMP:.*]] = insertvalue { i32, i32 } {{.*}}, i32 %[[#REAL_INT_CAST]], 0
// LLVM: %{{.*}} = insertvalue { i32, i32 } %[[TMP]], i32 %[[#IMAG_INT_CAST]], 1

void promotion() {
  cd = cf + cf;
}
