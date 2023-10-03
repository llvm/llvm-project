// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -DSME_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +bf16 -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +bf16 -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

#ifdef SME_OVERLOADED_FORMS
#define SME_ACLE_FUNC(A1,A2_UNUSED,A3) A1##A3
#else
#define SME_ACLE_FUNC(A1,A2,A3) A1##A2##A3
#endif

// CHECK-C-LABEL: @test_svmopa_za32_s8(
// CHECK-CXX-LABEL: @_Z19test_svmopa_za32_s8u10__SVBool_tu10__SVBool_tu10__SVInt8_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.smopa.wide.nxv16i8(i32 0, <vscale x 16 x i1> [[PN:%.*]], <vscale x 16 x i1> [[PM:%.*]], <vscale x 16 x i8> [[ZN:%.*]], <vscale x 16 x i8> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm) {
  SME_ACLE_FUNC(svmopa_za32, _s8, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za32_u8(
// CHECK-CXX-LABEL: @_Z19test_svmopa_za32_u8u10__SVBool_tu10__SVBool_tu11__SVUint8_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.umopa.wide.nxv16i8(i32 0, <vscale x 16 x i1> [[PN:%.*]], <vscale x 16 x i1> [[PM:%.*]], <vscale x 16 x i8> [[ZN:%.*]], <vscale x 16 x i8> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za32_u8(svbool_t pn, svbool_t pm, svuint8_t zn, svuint8_t zm) {
  SME_ACLE_FUNC(svmopa_za32, _u8, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za32_bf16(
// CHECK-CXX-LABEL: @_Z21test_svmopa_za32_bf16u10__SVBool_tu10__SVBool_tu14__SVBFloat16_tu14__SVBFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.mopa.wide.nxv8bf16(i32 0, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x bfloat> [[ZN:%.*]], <vscale x 8 x bfloat> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za32_bf16(svbool_t pn, svbool_t pm, svbfloat16_t zn, svbfloat16_t zm) {
  SME_ACLE_FUNC(svmopa_za32, _bf16, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za32_f16(
// CHECK-CXX-LABEL: @_Z20test_svmopa_za32_f16u10__SVBool_tu10__SVBool_tu13__SVFloat16_tu13__SVFloat16_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.mopa.wide.nxv8f16(i32 1, <vscale x 8 x i1> [[TMP0]], <vscale x 8 x i1> [[TMP1]], <vscale x 8 x half> [[ZN:%.*]], <vscale x 8 x half> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za32_f16(svbool_t pn, svbool_t pm, svfloat16_t zn, svfloat16_t zm) {
  SME_ACLE_FUNC(svmopa_za32, _f16, _m)(1, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svmopa_za32_f32(
// CHECK-CXX-LABEL: @_Z20test_svmopa_za32_f32u10__SVBool_tu10__SVBool_tu13__SVFloat32_tu13__SVFloat32_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PN:%.*]])
// CHECK-NEXT:    [[TMP1:%.*]] = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> [[PM:%.*]])
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> [[TMP0]], <vscale x 4 x i1> [[TMP1]], <vscale x 4 x float> [[ZN:%.*]], <vscale x 4 x float> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svmopa_za32_f32(svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm) {
  SME_ACLE_FUNC(svmopa_za32, _f32, _m)(1, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svsumopa_za32_s8(
// CHECK-CXX-LABEL: @_Z21test_svsumopa_za32_s8u10__SVBool_tu10__SVBool_tu10__SVInt8_tu11__SVUint8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.sumopa.wide.nxv16i8(i32 0, <vscale x 16 x i1> [[PN:%.*]], <vscale x 16 x i1> [[PM:%.*]], <vscale x 16 x i8> [[ZN:%.*]], <vscale x 16 x i8> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svsumopa_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svuint8_t zm) {
 SME_ACLE_FUNC(svsumopa_za32, _s8, _m)(0, pn, pm, zn, zm);
}

// CHECK-C-LABEL: @test_svusmopa_za32_u8(
// CHECK-CXX-LABEL: @_Z21test_svusmopa_za32_u8u10__SVBool_tu10__SVBool_tu11__SVUint8_tu10__SVInt8_t(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.usmopa.wide.nxv16i8(i32 0, <vscale x 16 x i1> [[PN:%.*]], <vscale x 16 x i1> [[PM:%.*]], <vscale x 16 x i8> [[ZN:%.*]], <vscale x 16 x i8> [[ZM:%.*]])
// CHECK-NEXT:    ret void
//
void test_svusmopa_za32_u8(svbool_t pn, svbool_t pm, svuint8_t zn, svint8_t zm) {
  SME_ACLE_FUNC(svusmopa_za32, _u8, _m)(0, pn, pm, zn, zm);
}
