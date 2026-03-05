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
//      CHECK-BEFORE:   %[[#R:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-BEFORE-NEXT:   %[[#I:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-BEFORE-NEXT:   %[[#C:]] = cir.complex.create %[[#R]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   cir.store{{.*}} %[[#C]], %{{.+}} : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#R:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-AFTER-NEXT:   %[[#I:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-AFTER-NEXT:   %[[#C:]] = cir.complex.create %[[#R]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   cir.store{{.*}} %[[#C]], %{{.+}} : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
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

// CHECK-BEFORE: cir.func
// CHECK-BEFORE: %{{.+}} = cir.const #cir.complex<#cir.fp<0.000000e+00> : !cir.double, #cir.fp<3.000000e+00> : !cir.double> : !cir.complex<!cir.double>
// CHECK-BEFORE: %{{.+}} = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<3> : !s32i> : !cir.complex<!s32i>
// CHECK-BEFORE: }

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
// CHECK-BEFORE-NEXT:   %[[#C2:]] = cir.load{{.*}} %[[#C2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   cir.store{{.*}} %[[#C2]], %[[#C_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#CI2_PTR:]] = cir.get_global @ci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#CI2:]] = cir.load{{.*}} %[[#CI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   cir.store{{.*}} %[[#CI2]], %[[#CI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
// CHECK-AFTER-NEXT:   %[[#C2_PTR:]] = cir.get_global @c2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#C2:]] = cir.load{{.*}} %[[#C2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   cir.store{{.*}} %[[#C2]], %[[#C_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#CI2_PTR:]] = cir.get_global @ci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#CI2:]] = cir.load{{.*}} %[[#CI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   cir.store{{.*}} %[[#CI2]], %[[#CI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
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
// CHECK-BEFORE-NEXT:   %[[#VC2:]] = cir.load volatile{{.*}} %[[#VC2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#VC_PTR:]] = cir.get_global @vc : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   cir.store volatile{{.*}} %[[#VC2]], %[[#VC_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[#VCI2_PTR:]] = cir.get_global @vci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[#VCI2:]] = cir.load volatile{{.*}} %[[#VCI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#VCI_PTR:]] = cir.get_global @vci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   cir.store volatile{{.*}} %[[#VCI2]], %[[#VCI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
// CHECK-AFTER-NEXT:   %[[#VC2_PTR:]] = cir.get_global @vc2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#VC2:]] = cir.load volatile{{.*}} %[[#VC2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#VC_PTR:]] = cir.get_global @vc : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   cir.store volatile{{.*}} %[[#VC2]], %[[#VC_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#VCI2_PTR:]] = cir.get_global @vci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[#VCI2:]] = cir.load volatile{{.*}} %[[#VCI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#VCI_PTR:]] = cir.get_global @vci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   cir.store volatile{{.*}} %[[#VCI2]], %[[#VCI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
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
// CHECK-BEFORE-NEXT:   %[[#B:]] = cir.load{{.*}} %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.real %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#B:]] = cir.load{{.*}} %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
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
// CHECK-BEFORE-NEXT:   %[[#B:]] = cir.load{{.*}} %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %{{.+}} = cir.complex.imag %[[#B]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#A:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[#B:]] = cir.load{{.*}} %[[#A]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
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
// CHECK-BEFORE-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#REAL:]] = cir.complex.real %[[COMPLEX]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#REAL:]] = cir.complex.real %[[COMPLEX]] : !cir.complex<!s32i> -> !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#REAL:]] = cir.complex.real %[[COMPLEX]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#REAL:]] = cir.complex.real %[[COMPLEX]] : !cir.complex<!s32i> -> !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local void @extract_real()
// LLVM:   %[[COMPLEX_D:.*]] = load { double, double }, ptr @c, align 8
// LLVM:   %[[R1:.*]] = extractvalue { double, double } %[[COMPLEX_D]], 0
// LLVM:   %[[COMPLEX_I:.*]] = load { i32, i32 }, ptr @ci, align 4
// LLVM:   %[[R2:.*]] = extractvalue { i32, i32 } %[[COMPLEX_I]], 0
// LLVM: }

int extract_real_and_add(int _Complex a, int _Complex b) {
  return __real__ a + __real__ b;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[REAL_A:.*]] = cir.complex.real %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CHECK-BEFORE-NEXT:   %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[REAL_B:.*]] = cir.complex.real %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CHECK-BEFORE-NEXT:   %[[ADD:.*]] = cir.binop(add, %[[REAL_A]], %[[REAL_B]]) nsw : !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[REAL_A:.*]] = cir.complex.real %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CHECK-AFTER-NEXT:   %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[REAL_B:.*]] = cir.complex.real %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CHECK-AFTER-NEXT:   %[[ADD:.*]] = cir.binop(add, %[[REAL_A]], %[[REAL_B]]) nsw : !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local i32 @extract_real_and_add
// LLVM:   %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM:   %[[REAL_A:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 0
// LLVM:   %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM:   %[[REAL_B:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 0
// LLVM:   %10 = add nsw i32 %[[REAL_A]], %[[REAL_B]]
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

// Note: GEP emitted by cir might not be the same as LLVM, due to constant folding.
// LLVM: define dso_local void @imag_ptr()
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), ptr %{{.+}}, align 8
// LLVM:   store ptr getelementptr inbounds nuw (i8, ptr @ci, i64 4), ptr %{{.+}}, align 8
// LLVM: }

void extract_imag() {
  double i1 = __imag__ c;
  int i2 = __imag__ ci;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-BEFORE-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-BEFORE-NEXT:   %[[#IMAG:]] = cir.complex.imag %[[COMPLEX]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-BEFORE:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-BEFORE-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[#IMAG:]] = cir.complex.imag %[[COMPLEX]] : !cir.complex<!s32i> -> !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-AFTER-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-AFTER-NEXT:   %[[#IMAG:]] = cir.complex.imag %[[COMPLEX]] : !cir.complex<!cir.double> -> !cir.double
//      CHECK-AFTER:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-AFTER-NEXT:   %[[COMPLEX:.*]] = cir.load{{.*}} %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[#IMAG:]] = cir.complex.imag %[[COMPLEX]] : !cir.complex<!s32i> -> !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local void @extract_imag()
// LLVM:   %[[COMPLEX_D:.*]] = load { double, double }, ptr @c, align 8
// LLVM:   %[[I1:.*]] = extractvalue { double, double } %[[COMPLEX_D]], 1
// LLVM:   %[[COMPLEX_I:.*]] = load { i32, i32 }, ptr @ci, align 4
// LLVM:   %[[I2:.*]] = extractvalue { i32, i32 } %[[COMPLEX_I]], 1
// LLVM: }

int extract_imag_and_add(int _Complex a, int _Complex b) {
  return __imag__ a + __imag__ b;
}

//      CHECK-BEFORE: cir.func
//      CHECK-BEFORE:   %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[IMAG_A:.*]] = cir.complex.imag %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CHECK-BEFORE-NEXT:   %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-BEFORE-NEXT:   %[[IMAG_B:.*]] = cir.complex.imag %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CHECK-BEFORE-NEXT:   %[[ADD:.*]] = cir.binop(add, %[[IMAG_A]], %[[IMAG_B]]) nsw : !s32i
//      CHECK-BEFORE: }

//      CHECK-AFTER: cir.func
//      CHECK-AFTER:   %[[COMPLEX_A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[IMAG_A:.*]] = cir.complex.imag %[[COMPLEX_A]] : !cir.complex<!s32i> -> !s32i
// CHECK-AFTER-NEXT:   %[[COMPLEX_B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-AFTER-NEXT:   %[[IMAG_B:.*]] = cir.complex.imag %[[COMPLEX_B]] : !cir.complex<!s32i> -> !s32i
// CHECK-AFTER-NEXT:   %[[ADD:.*]] = cir.binop(add, %[[IMAG_A]], %[[IMAG_B]]) nsw : !s32i
//      CHECK-AFTER: }

// LLVM: define dso_local i32 @extract_imag_and_add
// LLVM:   %[[COMPLEX_A:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM:   %[[IMAG_A:.*]] = extractvalue { i32, i32 } %[[COMPLEX_A]], 1
// LLVM:   %[[COMPLEX_B:.*]] = load { i32, i32 }, ptr {{.*}}, align 4
// LLVM:   %[[IMAG_B:.*]] = extractvalue { i32, i32 } %[[COMPLEX_B]], 1
// LLVM:   %10 = add nsw i32 %[[IMAG_A]], %[[IMAG_B]]
// LLVM: }

void complex_with_empty_init() { int _Complex c = {}; }

// CHECK: {{.*}} = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<0> : !s32i> : !cir.complex<!s32i>

void complex_array_subscript() {
  int _Complex arr[2];
  int _Complex r = arr[1];
}

// CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.complex<!s32i> x 2>, !cir.ptr<!cir.array<!cir.complex<!s32i> x 2>>, ["arr"]
// CHECK: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["r", init]
// CHECK: %[[IDX:.*]] = cir.const #cir.int<1> : !s32i
// CHECK: %[[RESULT_VAL:.*]] = cir.get_element %[[ARR]][%[[IDX]]] : (!cir.ptr<!cir.complex<!s32i>>, !s32i) -> !cir.ptr<!cir.complex<!s32i>>
// CHECK: %[[TMP:.*]] = cir.load{{.*}} %[[RESULT_VAL]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK: cir.store{{.*}} %[[TMP]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[ARR:.*]] = alloca [2 x { i32, i32 }], i64 1, align 16
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT_VAL:.*]] = getelementptr [2 x { i32, i32 }], ptr %[[ARR]], i32 0, i64 1
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[RESULT_VAL]], align 8
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[RESULT]], align 4
