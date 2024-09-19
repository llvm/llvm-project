// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CHECK-BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CHECK-BEFORE %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CHECK-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir -mmlir --mlir-print-ir-after=cir-canonicalize -o %t.cir %s 2>&1 | FileCheck --check-prefix=CHECK-AFTER %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --input-file=%t.ll --check-prefixes=LLVM %s

double _Complex c, c2;
int _Complex ci, ci2;

volatile double _Complex vc, vc2;
volatile int _Complex vci, vci2;

void list_init() {
  double _Complex c1 = {1.0, 2.0};
  int _Complex c2 = {1, 2};
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#REAL:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CHECK-BEFORE-NEXT:   %[[#IMAG:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>
//      CHECK-BEFORE:   %[[#REAL:]] = cir.const #cir.int<1> : !s32i
// CHECK-BEFORE-NEXT:   %[[#IMAG:]] = cir.const #cir.int<2> : !s32i
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>
//      CHECK-BEFORE: }

// CHECK-AFTER: cir.func
// CHECK-AFTER:   %{{.+}} = cir.const #cir.complex<#cir.fp<1.000000e+00> : !cir.double, #cir.fp<2.000000e+00> : !cir.double> : !cir.complex<!cir.double>
// CHECK-AFTER:   %{{.+}} = cir.const #cir.complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CHECK-AFTER: }

// LLVM: define dso_local void @list_init()
// LLVM:   store { double, double } { double 1.000000e+00, double 2.000000e+00 }, ptr %{{.+}}, align 8
// LLVM: }

void list_init_2(double r, double i) {
  double _Complex c1 = {r, i};
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#R:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-BEFORE-NEXT:   %[[#I:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-BEFORE-NEXT:   %[[#C:]] = cir.complex.create %[[#R]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   cir.store %[[#C]], %{{.+}} : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#R:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-AFTER-NEXT:   %[[#I:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-AFTER-NEXT:   %[[#C:]] = cir.complex.create %[[#R]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   cir.store %[[#C]], %{{.+}} : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
//      CHECK-AFTER: }

//      LLVM: define dso_local void @list_init_2(double %{{.+}}, double %{{.+}})
//      LLVM:   %[[#A:]] = insertvalue { double, double } undef, double %{{.+}}, 0
// LLVM-NEXT:   %[[#B:]] = insertvalue { double, double } %[[#A]], double %{{.+}}, 1
// LLVM-NEXT:   store { double, double } %[[#B]], ptr %5, align 8
//      LLVM: }

void builtin_init(double r, double i) {
  double _Complex c = __builtin_complex(r, i);
}

// CHECK-BEFORE: cir.func
// CHECK-BEFORE:   %{{.+}} = cir.complex.create %{{.+}}, %{{.+}} : !cir.double -> !cir.complex<!cir.double>
// CHECK-BEFORE: }

// CHECK-AFTER: cir.func
// CHECK-AFTER:   %{{.+}} = cir.complex.create %{{.+}}, %{{.+}} : !cir.double -> !cir.complex<!cir.double>
// CHECK-AFTER: }

//      LLVM: define dso_local void @builtin_init
//      LLVM:   %[[#A:]] = insertvalue { double, double } undef, double %{{.+}}, 0
// LLVM-NEXT:   %[[#B:]] = insertvalue { double, double } %[[#A]], double %{{.+}}, 1
// LLVM-NEXT:   store { double, double } %[[#B]], ptr %{{.+}}, align 8
//      LLVM: }

void imag_literal() {
  c = 3.0i;
  ci = 3i;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE: %[[#REAL:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CHECK-BEFORE-NEXT: %[[#IMAG:]] = cir.const #cir.fp<3.000000e+00> : !cir.double
// CHECK-BEFORE-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>
//      CHECK-BEFORE: %[[#REAL:]] = cir.const #cir.int<0> : !s32i
// CHECK-BEFORE-NEXT: %[[#IMAG:]] = cir.const #cir.int<3> : !s32i
// CHECK-BEFORE-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>
//      CHECK-BEFORE: }

// CHECK-AFTER: cir.func
// CHECK-AFTER:   %{{.+}} = cir.const #cir.complex<#cir.fp<0.000000e+00> : !cir.double, #cir.fp<3.000000e+00> : !cir.double> : !cir.complex<!cir.double>
// CHECK-AFTER:   %{{.+}} = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<3> : !s32i> : !cir.complex<!s32i>
// CHECK-AFTER: }

// LLVM: define dso_local void @imag_literal()
// LLVM:   store { double, double } { double 0.000000e+00, double 3.000000e+00 }, ptr @c, align 8
// LLVM:   store { i32, i32 } { i32 0, i32 3 }, ptr @ci, align 4
// LLVM: }

void load_store() {
  c = c2;
  ci = ci2;
}

//      CHECK-BEFORE: cir.func
// CHECK-BEFORE-NEXT:   %[[#C2_PTR:]] = cir.get_global @c2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#C2:]] = cir.load %[[#C2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   cir.store %[[#C2]], %[[#C_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#CI2_PTR:]] = cir.get_global @ci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#CI2:]] = cir.load %[[#CI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   cir.store %[[#CI2]], %[[#CI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
// CHECK-AFTER-NEXT:   %[[#C2_PTR:]] = cir.get_global @c2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#C2:]] = cir.load %[[#C2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   cir.store %[[#C2]], %[[#C_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#CI2_PTR:]] = cir.get_global @ci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#CI2:]] = cir.load %[[#CI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   cir.store %[[#CI2]], %[[#CI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-AFTER: }

//      LLVM: define dso_local void @load_store()
//      LLVM:   %[[#A:]] = load { double, double }, ptr @c2, align 8
// LLVM-NEXT:   store { double, double } %[[#A]], ptr @c, align 8
// LLVM-NEXT:   %[[#B:]] = load { i32, i32 }, ptr @ci2, align 4
// LLVM-NEXT:   store { i32, i32 } %[[#B]], ptr @ci, align 4
//      LLVM: }

void load_store_volatile() {
  vc = vc2;
  vci = vci2;
}

//      CHECK-BEFORE: cir.func
// CHECK-BEFORE-NEXT:   %[[#VC2_PTR:]] = cir.get_global @vc2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#VC2:]] = cir.load volatile %[[#VC2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#VC_PTR:]] = cir.get_global @vc : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   cir.store volatile %[[#VC2]], %[[#VC_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#VCI2_PTR:]] = cir.get_global @vci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#VCI2:]] = cir.load volatile %[[#VCI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#VCI_PTR:]] = cir.get_global @vci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   cir.store volatile %[[#VCI2]], %[[#VCI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
// CHECK-AFTER-NEXT:   %[[#VC2_PTR:]] = cir.get_global @vc2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#VC2:]] = cir.load volatile %[[#VC2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#VC_PTR:]] = cir.get_global @vc : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   cir.store volatile %[[#VC2]], %[[#VC_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#VCI2_PTR:]] = cir.get_global @vci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#VCI2:]] = cir.load volatile %[[#VCI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#VCI_PTR:]] = cir.get_global @vci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   cir.store volatile %[[#VCI2]], %[[#VCI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-AFTER: }

//      LLVM: define dso_local void @load_store_volatile()
//      LLVM:   %[[#A:]] = load volatile { double, double }, ptr @vc2, align 8
// LLVM-NEXT:   store volatile { double, double } %[[#A]], ptr @vc, align 8
// LLVM-NEXT:   %[[#B:]] = load volatile { i32, i32 }, ptr @vci2, align 4
// LLVM-NEXT:   store volatile { i32, i32 } %[[#B]], ptr @vci, align 4
//      LLVM: }

void real() {
  double r = __builtin_creal(c);
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.real %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.real %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-AFTER: }

//      LLVM: define dso_local void @real()
//      LLVM:   %[[#A:]] = extractvalue { double, double } %{{.+}}, 0
// LLVM-NEXT:   store double %[[#A]], ptr %{{.+}}, align 8
//      LLVM: }

void imag() {
  double i = __builtin_cimag(c);
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.imag %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.imag %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-AFTER: }

//      LLVM: define dso_local void @imag()
//      LLVM:   %[[#A:]] = extractvalue { double, double } %{{.+}}, 1
// LLVM-NEXT:   store double %[[#A]], ptr %{{.+}}, align 8
//      LLVM: }

void real_ptr() {
  double *r1 = &__real__ c;
  int *r2 = &__real__ ci;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK-AFTER: }

//      LLVM: define dso_local void @real_ptr()
//      LLVM:   store ptr @c, ptr %{{.+}}, align 8
// LLVM-NEXT:   store ptr @ci, ptr %{{.+}}, align 8
//      LLVM: }

void real_ptr_local() {
  double _Complex c1 = {1.0, 2.0};
  double *r3 = &__real__ c1;
}

// CHECK-BEFORE: cir.func
// CHECK-BEFORE:   %[[#C:]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE:   %{{.+}} = cir.complex.real_ptr %[[#C]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-BEFORE: }

// CHECK-AFTER: cir.func
// CHECK-AFTER:   %[[#C:]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER:   %{{.+}} = cir.complex.real_ptr %[[#C]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-AFTER: }

//      LLVM: define dso_local void @real_ptr_local()
//      LLVM:   store { double, double } { double 1.000000e+00, double 2.000000e+00 }, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.+}} = getelementptr inbounds { double, double }, ptr %{{.+}}, i32 0, i32 0
//      LLVM: }

void extract_real() {
  double r1 = __real__ c;
  int r2 = __real__ ci;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local void @extract_real()
// LLVM:   %{{.+}} = load double, ptr @c, align 8
// LLVM:   %{{.+}} = load i32, ptr @ci, align 4
// LLVM: }

void imag_ptr() {
  double *i1 = &__imag__ c;
  int *i2 = &__imag__ ci;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK-AFTER: }

// LLVM: define dso_local void @imag_ptr()
// LLVM:   store ptr getelementptr inbounds ({ double, double }, ptr @c, i32 0, i32 1), ptr %{{.+}}, align 8
// LLVM:   store ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1), ptr %{{.+}}, align 8
// LLVM: }

void extract_imag() {
  double i1 = __imag__ c;
  int i2 = __imag__ ci;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-AFTER-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local void @extract_imag()
// LLVM:   %{{.+}} = load double, ptr getelementptr inbounds ({ double, double }, ptr @c, i32 0, i32 1), align 8
// LLVM:   %{{.+}} = load i32, ptr getelementptr inbounds ({ i32, i32 }, ptr @ci, i32 0, i32 1), align 4
// LLVM: }
