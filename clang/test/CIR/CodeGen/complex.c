// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=C,CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --input-file=%t.cir --check-prefixes=CPP,CHECK %s

double _Complex c, c2;
int _Complex ci, ci2;

volatile double _Complex vc, vc2;
volatile int _Complex vci, vci2;

void list_init() {
  double _Complex c1 = {1.0, 2.0};
  int _Complex c2 = {1, 2};
}

//          C: cir.func no_proto @list_init()
//        CPP: cir.func @_Z9list_initv()
//      CHECK:   %[[#REAL:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
// CHECK-NEXT:   %[[#IMAG:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
// CHECK-NEXT:   %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>
//      CHECK:   %[[#REAL:]] = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:   %[[#IMAG:]] = cir.const #cir.int<2> : !s32i
// CHECK-NEXT:   %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>
//      CHECK: }

void list_init_2(double r, double i) {
  double _Complex c1 = {r, i};
}

//          C: cir.func @list_init_2
//        CPP: cir.func @_Z11list_init_2dd
//      CHECK:   %[[#R:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT:   %[[#I:]] = cir.load %{{.+}} : !cir.ptr<!cir.double>, !cir.double
// CHECK-NEXT:   %[[#C:]] = cir.complex.create %[[#R]], %[[#I]] : !cir.double -> !cir.complex<!cir.double>
// CHECK-NEXT:   cir.store %[[#C]], %{{.+}} : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
//      CHECK: }

void imag_literal() {
  c = 3.0i;
  ci = 3i;
}

//          C: cir.func no_proto @imag_literal()
//        CPP: cir.func @_Z12imag_literalv()
//      CHECK: %[[#REAL:]] = cir.const #cir.fp<0.000000e+00> : !cir.double
// CHECK-NEXT: %[[#IMAG:]] = cir.const #cir.fp<3.000000e+00> : !cir.double
// CHECK-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !cir.double -> !cir.complex<!cir.double>
//      CHECK: %[[#REAL:]] = cir.const #cir.int<0> : !s32i
// CHECK-NEXT: %[[#IMAG:]] = cir.const #cir.int<3> : !s32i
// CHECK-NEXT: %{{.+}} = cir.complex.create %[[#REAL]], %[[#IMAG]] : !s32i -> !cir.complex<!s32i>
//      CHECK: }

void load_store() {
  c = c2;
  ci = ci2;
}

//          C: cir.func no_proto @load_store()
//        CPP: cir.func @_Z10load_storev()
// CHECK-NEXT:   %[[#C2_PTR:]] = cir.get_global @c2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#C2:]] = cir.load %[[#C2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-NEXT:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   cir.store %[[#C2]], %[[#C_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#CI2_PTR:]] = cir.get_global @ci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %[[#CI2:]] = cir.load %[[#CI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-NEXT:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   cir.store %[[#CI2]], %[[#CI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK: }

void load_store_volatile() {
  vc = vc2;
  vci = vci2;
}

//          C: cir.func no_proto @load_store_volatile()
//        CPP: cir.func @_Z19load_store_volatilev()
// CHECK-NEXT:   %[[#VC2_PTR:]] = cir.get_global @vc2 : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#VC2:]] = cir.load volatile %[[#VC2_PTR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CHECK-NEXT:   %[[#VC_PTR:]] = cir.get_global @vc : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   cir.store volatile %[[#VC2]], %[[#VC_PTR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#VCI2_PTR:]] = cir.get_global @vci2 : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %[[#VCI2:]] = cir.load volatile %[[#VCI2_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CHECK-NEXT:   %[[#VCI_PTR:]] = cir.get_global @vci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   cir.store volatile %[[#VCI2]], %[[#VCI_PTR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
//      CHECK: }

void real_ptr() {
  double *r1 = &__real__ c;
  int *r2 = &__real__ ci;
}

//          C: cir.func no_proto @real_ptr()
//        CPP: cir.func @_Z8real_ptrv()
//      CHECK:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %{{.+}} = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK: }

void real_ptr_local() {
  double _Complex c1 = {1.0, 2.0};
  double *r3 = &__real__ c1;
}

//     C: cir.func no_proto @real_ptr_local()
//   CPP: cir.func @_Z14real_ptr_localv()
// CHECK:   %[[#C:]] = cir.alloca !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CHECK:   %{{.+}} = cir.complex.real_ptr %[[#C]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK: }

void extract_real() {
  double r1 = __real__ c;
  int r2 = __real__ ci;
}

//          C: cir.func no_proto @extract_real()
//        CPP: cir.func @_Z12extract_realv()
//      CHECK:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %[[#REAL_PTR:]] = cir.complex.real_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-NEXT:   %{{.+}} = cir.load %[[#REAL_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK: }

void imag_ptr() {
  double *i1 = &__imag__ c;
  int *i2 = &__imag__ ci;
}

//          C: cir.func no_proto @imag_ptr()
//        CPP: cir.func @_Z8imag_ptrv()
//      CHECK:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
//      CHECK:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %{{.+}} = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
//      CHECK: }

void extract_imag() {
  double i1 = __imag__ c;
  int i2 = __imag__ ci;
}

//          C: cir.func no_proto @extract_imag()
//        CPP: cir.func @_Z12extract_imagv()
//      CHECK:   %[[#C_PTR:]] = cir.get_global @c : !cir.ptr<!cir.complex<!cir.double>>
// CHECK-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#C_PTR]] : !cir.ptr<!cir.complex<!cir.double>> -> !cir.ptr<!cir.double>
// CHECK-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!cir.double>, !cir.double
//      CHECK:   %[[#CI_PTR:]] = cir.get_global @ci : !cir.ptr<!cir.complex<!s32i>>
// CHECK-NEXT:   %[[#IMAG_PTR:]] = cir.complex.imag_ptr %[[#CI_PTR]] : !cir.ptr<!cir.complex<!s32i>> -> !cir.ptr<!s32i>
// CHECK-NEXT:   %{{.+}} = cir.load %[[#IMAG_PTR]] : !cir.ptr<!s32i>, !s32i
//      CHECK: }
