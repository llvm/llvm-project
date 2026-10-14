// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=OGCG %s

int g;
int &rg = g;
int ref_only() { return rg; }

// CIR-LABEL: cir.func {{.*}} @_Z8ref_onlyv()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[G:.*]] = cir.const #cir.global_view<@g> : !cir.ptr<!s32i>
// CIR-NEXT:    %[[VAL:.*]] = cir.load align(4) %[[G]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    cir.store %[[VAL]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} i32 @_Z8ref_onlyv()
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM-NEXT:    %[[VAL:.*]] = load i32, ptr @g, align 4
// LLVM-NEXT:    store i32 %[[VAL]], ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} i32 @_Z8ref_onlyv()
// OGCG:         %[[VAL:.*]] = load i32, ptr @g, align 4
// OGCG-NEXT:    ret i32 %[[VAL]]

int g2;
const int &crg = g2;
int value_or_ref_as_ref() { return crg; }

// CIR-LABEL: cir.func {{.*}} @_Z19value_or_ref_as_refv()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[G2:.*]] = cir.const #cir.global_view<@g2> : !cir.ptr<!s32i>
// CIR-NEXT:    %[[VAL:.*]] = cir.load align(4) %[[G2]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    cir.store %[[VAL]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} i32 @_Z19value_or_ref_as_refv()
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM-NEXT:    %[[VAL:.*]] = load i32, ptr @g2, align 4
// LLVM-NEXT:    store i32 %[[VAL]], ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} i32 @_Z19value_or_ref_as_refv()
// OGCG:         %[[VAL:.*]] = load i32, ptr @g2, align 4
// OGCG-NEXT:    ret i32 %[[VAL]]

const int k = 7;
const int &crk = k;
int value_or_ref_as_value() { return crk; }

// CIR-LABEL: cir.func {{.*}} @_Z21value_or_ref_as_valuev()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[SEVEN:.*]] = cir.const #cir.int<7> : !s32i
// CIR-NEXT:    cir.store %[[SEVEN]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} i32 @_Z21value_or_ref_as_valuev()
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM-NEXT:    store i32 7, ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} i32 @_Z21value_or_ref_as_valuev()
// OGCG:         ret i32 7

struct S { static const int sm = 9; };
int static_member() { return S::sm; }

// CIR-LABEL: cir.func {{.*}} @_Z13static_memberv()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[NINE:.*]] = cir.const #cir.int<9> : !s32i
// CIR-NEXT:    cir.store %[[NINE]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} i32 @_Z13static_memberv()
// LLVM:         %[[RETVAL:.*]] = alloca i32
// LLVM-NEXT:    store i32 9, ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} i32 @_Z13static_memberv()
// OGCG:         ret i32 9

const _Complex float cc = 1.0f;
_Complex float complex_value() { return cc; }

// CIR-LABEL: cir.func {{.*}} @_Z13complex_valuev()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[CPLX:.*]] = cir.const #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float, #cir.fp<0.000000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR-NEXT:    cir.store align(4) %[[CPLX]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} <2 x float> @_Z13complex_valuev()
// LLVM:         %[[RETVAL:.*]] = alloca { float, float }, align 4
// LLVM-NEXT:    store { float, float } { float 1.000000e+00, float 0.000000e+00 }, ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} <2 x float> @_Z13complex_valuev()
// OGCG:         %[[REALP:.*]] = getelementptr inbounds nuw { float, float }, ptr %{{.*}}, i32 0, i32 0
// OGCG-NEXT:    %[[IMAGP:.*]] = getelementptr inbounds nuw { float, float }, ptr %{{.*}}, i32 0, i32 1
// OGCG-NEXT:    store float 1.000000e+00, ptr %[[REALP]]
// OGCG-NEXT:    store float 0.000000e+00, ptr %[[IMAGP]]

_Complex float cg;
_Complex float &rcg = cg;
_Complex float complex_ref() { return rcg; }

// CIR-LABEL: cir.func {{.*}} @_Z11complex_refv()
// CIR:         %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR-NEXT:    %[[CG:.*]] = cir.const #cir.global_view<@cg> : !cir.ptr<!cir.complex<!cir.float>>
// CIR-NEXT:    %[[VAL:.*]] = cir.load align(4) %[[CG]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR-NEXT:    cir.store align(4) %[[VAL]], %[[RETVAL]]

// LLVM-LABEL: define {{.*}} <2 x float> @_Z11complex_refv()
// LLVM:         %[[RETVAL:.*]] = alloca { float, float }, align 4
// LLVM-NEXT:    %[[VAL:.*]] = load { float, float }, ptr @cg, align 4
// LLVM-NEXT:    store { float, float } %[[VAL]], ptr %[[RETVAL]]

// OGCG-LABEL: define {{.*}} <2 x float> @_Z11complex_refv()
// OGCG:         %{{.*}} = load float, ptr @cg, align 4
// OGCG-NEXT:    %{{.*}} = load float, ptr getelementptr inbounds nuw (i8, ptr @cg, i64 4), align 4
