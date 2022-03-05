// REQUIRES: ve-registered-target

// RUN: %clang_cc1 -S -emit-llvm -triple ve-unknown-linux-gnu \
// RUN:   -ffreestanding %s -o - | FileCheck %s

#include <velintrin.h>

__vr vr1;

void __attribute__((noinline))
test_vld_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vld_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldu_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldunc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldunc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldunc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldunc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldunc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldunc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldunc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldunc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlsx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldlsx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlsx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsx.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlsx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlsxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldlsxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlsxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsxnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlsxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlzx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldlzx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlzx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzx.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlzx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlzxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldlzxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlzxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzxnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlzxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vld2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2d_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2d.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vld2d_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld2d_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2d_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2d.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld2d_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vld2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2dnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vld2dnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld2dnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2dnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2dnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld2dnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2d_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldu2d_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu2d_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2d_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2d.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu2d_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2dnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldu2dnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu2dnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2dnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2dnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu2dnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dsx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dsx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsx.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dsxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dsxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsxnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dzx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dzx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzx.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dzxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dzxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzxnc.vssvl(i64 %{{.*}}, i8* %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzxnc_vssvl(idx, p, vr1, 256);
}
