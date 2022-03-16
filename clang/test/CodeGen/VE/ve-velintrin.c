// REQUIRES: ve-registered-target

// RUN: %clang_cc1 -S -emit-llvm -triple ve-unknown-linux-gnu \
// RUN:   -ffreestanding %s -o - | FileCheck %s

#include <velintrin.h>

long v1, v2, v3;
double vd1;
float vf1;
__vr vr1, vr2, vr3, vr4;

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

void __attribute__((noinline))
test_vst_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst_vssl
  // CHECK: call void @llvm.ve.vl.vst.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vst_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstnc_vssl
  // CHECK: call void @llvm.ve.vl.vstnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstot_vssl
  // CHECK: call void @llvm.ve.vl.vstot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstncot_vssl
  // CHECK: call void @llvm.ve.vl.vstncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu_vssl
  // CHECK: call void @llvm.ve.vl.vstu.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstu_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstunc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstunc_vssl
  // CHECK: call void @llvm.ve.vl.vstunc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstunc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstuot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstuot_vssl
  // CHECK: call void @llvm.ve.vl.vstuot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstuot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstuncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstuncot_vssl
  // CHECK: call void @llvm.ve.vl.vstuncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstuncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl_vssl
  // CHECK: call void @llvm.ve.vl.vstl.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstl_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlnc_vssl
  // CHECK: call void @llvm.ve.vl.vstlnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstlnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlot_vssl
  // CHECK: call void @llvm.ve.vl.vstlot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstlot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlncot_vssl
  // CHECK: call void @llvm.ve.vl.vstlncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstlncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2d_vssl
  // CHECK: call void @llvm.ve.vl.vst2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vst2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vst2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vst2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dot_vssl
  // CHECK: call void @llvm.ve.vl.vst2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vst2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vst2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vst2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2d_vssl
  // CHECK: call void @llvm.ve.vl.vstu2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstu2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstu2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dot_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstu2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstu2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2d_vssl
  // CHECK: call void @llvm.ve.vl.vstl2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstl2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstl2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dot_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstl2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_vstl2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_pfchv_ssl(char* p, long idx) {
  // CHECK-LABEL: @test_pfchv_ssl
  // CHECK: call void @llvm.ve.vl.pfchv.ssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_pfchv_ssl(idx, p, 256);
}

void __attribute__((noinline))
test_pfchvnc_ssl(char* p, long idx) {
  // CHECK-LABEL: @test_pfchvnc_ssl
  // CHECK: call void @llvm.ve.vl.pfchvnc.ssl(i64 %{{.*}}, i8* %{{.*}}, i32 256)
  _vel_pfchvnc_ssl(idx, p, 256);
}

void __attribute__((noinline))
test_lsv_vvss(int idx) {
  // CHECK-LABEL: @test_lsv_vvss
  // CHECK: call <256 x double> @llvm.ve.vl.lsv.vvss(<256 x double> %{{.*}}, i32 %{{.*}}, i64 %{{.*}})
  vr1 = _vel_lsv_vvss(vr1, idx, v1);
}

void __attribute__((noinline))
test_lvsl_svs(int idx) {
  // CHECK-LABEL: @test_lvsl_svs
  // CHECK: call i64 @llvm.ve.vl.lvsl.svs(<256 x double> %{{.*}}, i32 %{{.*}})
  v1 = _vel_lvsl_svs(vr1, idx);
}

void __attribute__((noinline))
test_lvsd_svs(int idx) {
  // CHECK-LABEL: @test_lvsd_svs
  // CHECK: call double @llvm.ve.vl.lvsd.svs(<256 x double> %{{.*}}, i32 %{{.*}})
  vd1 = _vel_lvsd_svs(vr1, idx);
}

void __attribute__((noinline))
test_lvss_svs(int idx) {
  // CHECK-LABEL: @test_lvss_svs
  // CHECK: call float @llvm.ve.vl.lvss.svs(<256 x double> %{{.*}}, i32 %{{.*}})
  vf1 = _vel_lvss_svs(vr1, idx);
}

void __attribute__((noinline))
test_vbrdd_vsl() {
  // CHECK-LABEL: @test_vbrdd_vsl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdd.vsl(double %{{.*}}, i32 256)
  vr1 = _vel_vbrdd_vsl(vd1, 256);
}

void __attribute__((noinline))
test_vbrdd_vsvl() {
  // CHECK-LABEL: @test_vbrdd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdd_vsvl(vd1, vr1, 256);
}

void __attribute__((noinline))
test_vbrdl_vsl() {
  // CHECK-LABEL: @test_vbrdl_vsl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdl.vsl(i64 %{{.*}}, i32 256)
  vr1 = _vel_vbrdl_vsl(v1, 256);
}

void __attribute__((noinline))
test_vbrdl_vsvl() {
  // CHECK-LABEL: @test_vbrdl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdl_vsvl(v1, vr1, 256);
}

void __attribute__((noinline))
test_vbrds_vsl() {
  // CHECK-LABEL: @test_vbrds_vsl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrds.vsl(float %{{.*}}, i32 256)
  vr1 = _vel_vbrds_vsl(vf1, 256);
}

void __attribute__((noinline))
test_vbrds_vsvl() {
  // CHECK-LABEL: @test_vbrds_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrds.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrds_vsvl(vf1, vr1, 256);
}

void __attribute__((noinline))
test_vbrdw_vsl() {
  // CHECK-LABEL: @test_vbrdw_vsl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdw.vsl(i32 %{{.*}}, i32 256)
  vr1 = _vel_vbrdw_vsl(v1, 256);
}

void __attribute__((noinline))
test_vbrdw_vsvl() {
  // CHECK-LABEL: @test_vbrdw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdw_vsvl(v1, vr1, 256);
}

void __attribute__((noinline))
test_pvbrd_vsl() {
  // CHECK-LABEL: @test_pvbrd_vsl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrd.vsl(i64 %{{.*}}, i32 256)
  vr1 = _vel_pvbrd_vsl(v1, 256);
}

void __attribute__((noinline))
test_pvbrd_vsvl() {
  // CHECK-LABEL: @test_pvbrd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrd.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_pvbrd_vsvl(v1, vr1, 256);
}

void __attribute__((noinline))
test_vmv_vsvl() {
  // CHECK-LABEL: @test_vmv_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmv.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vmv_vsvl(v1, vr1, 256);
}

void __attribute__((noinline))
test_vmv_vsvvl() {
  // CHECK-LABEL: @test_vmv_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmv.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vmv_vsvvl(v1, vr1, vr2, 256);
}

void __attribute__((noinline))
test_vaddul_vvvl() {
  // CHECK-LABEL: @test_vaddul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vaddul_vvvvl() {
  // CHECK-LABEL: @test_vaddul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddul_vsvl() {
  // CHECK-LABEL: @test_vaddul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vaddul_vsvvl() {
  // CHECK-LABEL: @test_vaddul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vadduw_vvvl() {
  // CHECK-LABEL: @test_vadduw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vadduw_vvvvl() {
  // CHECK-LABEL: @test_vadduw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vadduw_vsvl() {
  // CHECK-LABEL: @test_vadduw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vadduw_vsvvl() {
  // CHECK-LABEL: @test_vadduw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvaddu_vvvl() {
  // CHECK-LABEL: @test_pvaddu_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvaddu_vvvvl() {
  // CHECK-LABEL: @test_pvaddu_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvaddu_vsvl() {
  // CHECK-LABEL: @test_pvaddu_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvaddu_vsvvl() {
  // CHECK-LABEL: @test_pvaddu_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddswsx_vvvl() {
  // CHECK-LABEL: @test_vaddswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vaddswsx_vvvvl() {
  // CHECK-LABEL: @test_vaddswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddswsx_vsvl() {
  // CHECK-LABEL: @test_vaddswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vaddswsx_vsvvl() {
  // CHECK-LABEL: @test_vaddswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddswzx_vvvl() {
  // CHECK-LABEL: @test_vaddswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vaddswzx_vvvvl() {
  // CHECK-LABEL: @test_vaddswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddswzx_vsvl() {
  // CHECK-LABEL: @test_vaddswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vaddswzx_vsvvl() {
  // CHECK-LABEL: @test_vaddswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvadds_vvvl() {
  // CHECK-LABEL: @test_pvadds_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvadds_vvvvl() {
  // CHECK-LABEL: @test_pvadds_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvadds_vsvl() {
  // CHECK-LABEL: @test_pvadds_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvadds_vsvvl() {
  // CHECK-LABEL: @test_pvadds_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddsl_vvvl() {
  // CHECK-LABEL: @test_vaddsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vaddsl_vvvvl() {
  // CHECK-LABEL: @test_vaddsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vaddsl_vsvl() {
  // CHECK-LABEL: @test_vaddsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vaddsl_vsvvl() {
  // CHECK-LABEL: @test_vaddsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubul_vvvl() {
  // CHECK-LABEL: @test_vsubul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsubul_vvvvl() {
  // CHECK-LABEL: @test_vsubul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubul_vsvl() {
  // CHECK-LABEL: @test_vsubul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vsubul_vsvvl() {
  // CHECK-LABEL: @test_vsubul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubuw_vvvl() {
  // CHECK-LABEL: @test_vsubuw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsubuw_vvvvl() {
  // CHECK-LABEL: @test_vsubuw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubuw_vsvl() {
  // CHECK-LABEL: @test_vsubuw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vsubuw_vsvvl() {
  // CHECK-LABEL: @test_vsubuw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsubu_vvvl() {
  // CHECK-LABEL: @test_pvsubu_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsubu_vvvvl() {
  // CHECK-LABEL: @test_pvsubu_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsubu_vsvl() {
  // CHECK-LABEL: @test_pvsubu_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvsubu_vsvvl() {
  // CHECK-LABEL: @test_pvsubu_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubswsx_vvvl() {
  // CHECK-LABEL: @test_vsubswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsubswsx_vvvvl() {
  // CHECK-LABEL: @test_vsubswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubswsx_vsvl() {
  // CHECK-LABEL: @test_vsubswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vsubswsx_vsvvl() {
  // CHECK-LABEL: @test_vsubswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubswzx_vvvl() {
  // CHECK-LABEL: @test_vsubswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsubswzx_vvvvl() {
  // CHECK-LABEL: @test_vsubswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubswzx_vsvl() {
  // CHECK-LABEL: @test_vsubswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vsubswzx_vsvvl() {
  // CHECK-LABEL: @test_vsubswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsubs_vvvl() {
  // CHECK-LABEL: @test_pvsubs_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsubs_vvvvl() {
  // CHECK-LABEL: @test_pvsubs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsubs_vsvl() {
  // CHECK-LABEL: @test_pvsubs_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvsubs_vsvvl() {
  // CHECK-LABEL: @test_pvsubs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubsl_vvvl() {
  // CHECK-LABEL: @test_vsubsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsubsl_vvvvl() {
  // CHECK-LABEL: @test_vsubsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsubsl_vsvl() {
  // CHECK-LABEL: @test_vsubsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vsubsl_vsvvl() {
  // CHECK-LABEL: @test_vsubsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulul_vvvl() {
  // CHECK-LABEL: @test_vmulul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmulul_vvvvl() {
  // CHECK-LABEL: @test_vmulul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulul_vsvl() {
  // CHECK-LABEL: @test_vmulul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmulul_vsvvl() {
  // CHECK-LABEL: @test_vmulul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmuluw_vvvl() {
  // CHECK-LABEL: @test_vmuluw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmuluw_vvvvl() {
  // CHECK-LABEL: @test_vmuluw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmuluw_vsvl() {
  // CHECK-LABEL: @test_vmuluw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmuluw_vsvvl() {
  // CHECK-LABEL: @test_vmuluw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulswsx_vvvl() {
  // CHECK-LABEL: @test_vmulswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmulswsx_vvvvl() {
  // CHECK-LABEL: @test_vmulswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulswsx_vsvl() {
  // CHECK-LABEL: @test_vmulswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmulswsx_vsvvl() {
  // CHECK-LABEL: @test_vmulswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulswzx_vvvl() {
  // CHECK-LABEL: @test_vmulswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmulswzx_vvvvl() {
  // CHECK-LABEL: @test_vmulswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulswzx_vsvl() {
  // CHECK-LABEL: @test_vmulswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmulswzx_vsvvl() {
  // CHECK-LABEL: @test_vmulswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulsl_vvvl() {
  // CHECK-LABEL: @test_vmulsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmulsl_vvvvl() {
  // CHECK-LABEL: @test_vmulsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulsl_vsvl() {
  // CHECK-LABEL: @test_vmulsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmulsl_vsvvl() {
  // CHECK-LABEL: @test_vmulsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulslw_vvvl() {
  // CHECK-LABEL: @test_vmulslw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulslw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulslw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmulslw_vvvvl() {
  // CHECK-LABEL: @test_vmulslw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulslw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulslw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmulslw_vsvl() {
  // CHECK-LABEL: @test_vmulslw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulslw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulslw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmulslw_vsvvl() {
  // CHECK-LABEL: @test_vmulslw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulslw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulslw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivul_vvvl() {
  // CHECK-LABEL: @test_vdivul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vdivul_vvvvl() {
  // CHECK-LABEL: @test_vdivul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivul_vsvl() {
  // CHECK-LABEL: @test_vdivul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vdivul_vsvvl() {
  // CHECK-LABEL: @test_vdivul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivuw_vvvl() {
  // CHECK-LABEL: @test_vdivuw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vdivuw_vvvvl() {
  // CHECK-LABEL: @test_vdivuw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivuw_vsvl() {
  // CHECK-LABEL: @test_vdivuw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vdivuw_vsvvl() {
  // CHECK-LABEL: @test_vdivuw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivul_vvsl() {
  // CHECK-LABEL: @test_vdivul_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vdivul_vvsvl() {
  // CHECK-LABEL: @test_vdivul_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vdivuw_vvsl() {
  // CHECK-LABEL: @test_vdivuw_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vdivuw_vvsvl() {
  // CHECK-LABEL: @test_vdivuw_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswsx_vvvl() {
  // CHECK-LABEL: @test_vdivswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vdivswsx_vvvvl() {
  // CHECK-LABEL: @test_vdivswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswsx_vsvl() {
  // CHECK-LABEL: @test_vdivswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vdivswsx_vsvvl() {
  // CHECK-LABEL: @test_vdivswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswzx_vvvl() {
  // CHECK-LABEL: @test_vdivswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vdivswzx_vvvvl() {
  // CHECK-LABEL: @test_vdivswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswzx_vsvl() {
  // CHECK-LABEL: @test_vdivswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vdivswzx_vsvvl() {
  // CHECK-LABEL: @test_vdivswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswsx_vvsl() {
  // CHECK-LABEL: @test_vdivswsx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vdivswsx_vvsvl() {
  // CHECK-LABEL: @test_vdivswsx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vdivswzx_vvsl() {
  // CHECK-LABEL: @test_vdivswzx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vdivswzx_vvsvl() {
  // CHECK-LABEL: @test_vdivswzx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vdivsl_vvvl() {
  // CHECK-LABEL: @test_vdivsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vdivsl_vvvvl() {
  // CHECK-LABEL: @test_vdivsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivsl_vsvl() {
  // CHECK-LABEL: @test_vdivsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vdivsl_vsvvl() {
  // CHECK-LABEL: @test_vdivsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vdivsl_vvsl() {
  // CHECK-LABEL: @test_vdivsl_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vdivsl_vvsvl() {
  // CHECK-LABEL: @test_vdivsl_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpul_vvvl() {
  // CHECK-LABEL: @test_vcmpul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpul_vvvvl() {
  // CHECK-LABEL: @test_vcmpul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpul_vsvl() {
  // CHECK-LABEL: @test_vcmpul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpul_vsvvl() {
  // CHECK-LABEL: @test_vcmpul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpuw_vvvl() {
  // CHECK-LABEL: @test_vcmpuw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpuw_vvvvl() {
  // CHECK-LABEL: @test_vcmpuw_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpuw_vsvl() {
  // CHECK-LABEL: @test_vcmpuw_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpuw_vsvvl() {
  // CHECK-LABEL: @test_vcmpuw_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvcmpu_vvvl() {
  // CHECK-LABEL: @test_pvcmpu_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvcmpu_vvvvl() {
  // CHECK-LABEL: @test_pvcmpu_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvcmpu_vsvl() {
  // CHECK-LABEL: @test_pvcmpu_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvcmpu_vsvvl() {
  // CHECK-LABEL: @test_pvcmpu_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswsx_vvvl() {
  // CHECK-LABEL: @test_vcmpswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpswsx_vvvvl() {
  // CHECK-LABEL: @test_vcmpswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswsx_vsvl() {
  // CHECK-LABEL: @test_vcmpswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpswsx_vsvvl() {
  // CHECK-LABEL: @test_vcmpswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswzx_vvvl() {
  // CHECK-LABEL: @test_vcmpswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpswzx_vvvvl() {
  // CHECK-LABEL: @test_vcmpswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswzx_vsvl() {
  // CHECK-LABEL: @test_vcmpswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpswzx_vsvvl() {
  // CHECK-LABEL: @test_vcmpswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvcmps_vvvl() {
  // CHECK-LABEL: @test_pvcmps_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvcmps_vvvvl() {
  // CHECK-LABEL: @test_pvcmps_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvcmps_vsvl() {
  // CHECK-LABEL: @test_pvcmps_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvcmps_vsvvl() {
  // CHECK-LABEL: @test_pvcmps_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpsl_vvvl() {
  // CHECK-LABEL: @test_vcmpsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpsl_vvvvl() {
  // CHECK-LABEL: @test_vcmpsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vcmpsl_vsvl() {
  // CHECK-LABEL: @test_vcmpsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vcmpsl_vsvvl() {
  // CHECK-LABEL: @test_vcmpsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswsx_vvvl() {
  // CHECK-LABEL: @test_vmaxswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxswsx_vvvvl() {
  // CHECK-LABEL: @test_vmaxswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswsx_vsvl() {
  // CHECK-LABEL: @test_vmaxswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxswsx_vsvvl() {
  // CHECK-LABEL: @test_vmaxswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswzx_vvvl() {
  // CHECK-LABEL: @test_vmaxswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxswzx_vvvvl() {
  // CHECK-LABEL: @test_vmaxswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswzx_vsvl() {
  // CHECK-LABEL: @test_vmaxswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxswzx_vsvvl() {
  // CHECK-LABEL: @test_vmaxswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvmaxs_vvvl() {
  // CHECK-LABEL: @test_pvmaxs_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvmaxs_vvvvl() {
  // CHECK-LABEL: @test_pvmaxs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvmaxs_vsvl() {
  // CHECK-LABEL: @test_pvmaxs_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvmaxs_vsvvl() {
  // CHECK-LABEL: @test_pvmaxs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminswsx_vvvl() {
  // CHECK-LABEL: @test_vminswsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vminswsx_vvvvl() {
  // CHECK-LABEL: @test_vminswsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminswsx_vsvl() {
  // CHECK-LABEL: @test_vminswsx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vminswsx_vsvvl() {
  // CHECK-LABEL: @test_vminswsx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminswzx_vvvl() {
  // CHECK-LABEL: @test_vminswzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vminswzx_vvvvl() {
  // CHECK-LABEL: @test_vminswzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminswzx_vsvl() {
  // CHECK-LABEL: @test_vminswzx_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vsvl(i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vminswzx_vsvvl() {
  // CHECK-LABEL: @test_vminswzx_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vsvvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvmins_vvvl() {
  // CHECK-LABEL: @test_pvmins_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvmins_vvvvl() {
  // CHECK-LABEL: @test_pvmins_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvmins_vsvl() {
  // CHECK-LABEL: @test_pvmins_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvmins_vsvvl() {
  // CHECK-LABEL: @test_pvmins_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxsl_vvvl() {
  // CHECK-LABEL: @test_vmaxsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxsl_vvvvl() {
  // CHECK-LABEL: @test_vmaxsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vmaxsl_vsvl() {
  // CHECK-LABEL: @test_vmaxsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vmaxsl_vsvvl() {
  // CHECK-LABEL: @test_vmaxsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminsl_vvvl() {
  // CHECK-LABEL: @test_vminsl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vminsl_vvvvl() {
  // CHECK-LABEL: @test_vminsl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vminsl_vsvl() {
  // CHECK-LABEL: @test_vminsl_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vminsl_vsvvl() {
  // CHECK-LABEL: @test_vminsl_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vand_vvvl() {
  // CHECK-LABEL: @test_vand_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vand_vvvvl() {
  // CHECK-LABEL: @test_vand_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vand_vsvl() {
  // CHECK-LABEL: @test_vand_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vand_vsvvl() {
  // CHECK-LABEL: @test_vand_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvand_vvvl() {
  // CHECK-LABEL: @test_pvand_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvand_vvvvl() {
  // CHECK-LABEL: @test_pvand_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvand_vsvl() {
  // CHECK-LABEL: @test_pvand_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvand_vsvvl() {
  // CHECK-LABEL: @test_pvand_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vor_vvvl() {
  // CHECK-LABEL: @test_vor_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vor_vvvvl() {
  // CHECK-LABEL: @test_vor_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vor_vsvl() {
  // CHECK-LABEL: @test_vor_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vor_vsvvl() {
  // CHECK-LABEL: @test_vor_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvor_vvvl() {
  // CHECK-LABEL: @test_pvor_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvor_vvvvl() {
  // CHECK-LABEL: @test_pvor_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvor_vsvl() {
  // CHECK-LABEL: @test_pvor_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvor_vsvvl() {
  // CHECK-LABEL: @test_pvor_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vxor_vvvl() {
  // CHECK-LABEL: @test_vxor_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vxor_vvvvl() {
  // CHECK-LABEL: @test_vxor_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vxor_vsvl() {
  // CHECK-LABEL: @test_vxor_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_vxor_vsvvl() {
  // CHECK-LABEL: @test_vxor_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvxor_vvvl() {
  // CHECK-LABEL: @test_pvxor_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvxor_vvvvl() {
  // CHECK-LABEL: @test_pvxor_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvxor_vsvl() {
  // CHECK-LABEL: @test_pvxor_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvxor_vsvvl() {
  // CHECK-LABEL: @test_pvxor_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_veqv_vvvl() {
  // CHECK-LABEL: @test_veqv_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_veqv_vvvvl() {
  // CHECK-LABEL: @test_veqv_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_veqv_vsvl() {
  // CHECK-LABEL: @test_veqv_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_veqv_vsvvl() {
  // CHECK-LABEL: @test_veqv_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pveqv_vvvl() {
  // CHECK-LABEL: @test_pveqv_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pveqv_vvvvl() {
  // CHECK-LABEL: @test_pveqv_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pveqv_vsvl() {
  // CHECK-LABEL: @test_pveqv_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pveqv_vsvvl() {
  // CHECK-LABEL: @test_pveqv_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vldz_vvl() {
  // CHECK-LABEL: @test_vldz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldz.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vldz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vldz_vvvl() {
  // CHECK-LABEL: @test_vldz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vldz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvldzlo_vvl() {
  // CHECK-LABEL: @test_pvldzlo_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzlo.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzlo_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvldzlo_vvvl() {
  // CHECK-LABEL: @test_pvldzlo_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzlo.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzlo_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvldzup_vvl() {
  // CHECK-LABEL: @test_pvldzup_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzup.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzup_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvldzup_vvvl() {
  // CHECK-LABEL: @test_pvldzup_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzup.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzup_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvldz_vvl() {
  // CHECK-LABEL: @test_pvldz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldz.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvldz_vvvl() {
  // CHECK-LABEL: @test_pvldz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vpcnt_vvl() {
  // CHECK-LABEL: @test_vpcnt_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vpcnt.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vpcnt_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vpcnt_vvvl() {
  // CHECK-LABEL: @test_vpcnt_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vpcnt.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vpcnt_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvpcntlo_vvl() {
  // CHECK-LABEL: @test_pvpcntlo_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntlo.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntlo_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvpcntlo_vvvl() {
  // CHECK-LABEL: @test_pvpcntlo_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntlo.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntlo_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvpcntup_vvl() {
  // CHECK-LABEL: @test_pvpcntup_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntup.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntup_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvpcntup_vvvl() {
  // CHECK-LABEL: @test_pvpcntup_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntup.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntup_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvpcnt_vvl() {
  // CHECK-LABEL: @test_pvpcnt_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcnt.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcnt_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvpcnt_vvvl() {
  // CHECK-LABEL: @test_pvpcnt_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcnt.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcnt_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vbrv_vvl() {
  // CHECK-LABEL: @test_vbrv_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrv.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vbrv_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vbrv_vvvl() {
  // CHECK-LABEL: @test_vbrv_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrv.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vbrv_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvbrvlo_vvl() {
  // CHECK-LABEL: @test_pvbrvlo_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvlo.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvlo_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvbrvlo_vvvl() {
  // CHECK-LABEL: @test_pvbrvlo_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvlo.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvlo_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvbrvup_vvl() {
  // CHECK-LABEL: @test_pvbrvup_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvup.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvup_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvbrvup_vvvl() {
  // CHECK-LABEL: @test_pvbrvup_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvup.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvup_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvbrv_vvl() {
  // CHECK-LABEL: @test_pvbrv_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrv.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrv_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvbrv_vvvl() {
  // CHECK-LABEL: @test_pvbrv_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrv.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrv_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vseq_vl() {
  // CHECK-LABEL: @test_vseq_vl
  // CHECK: call <256 x double> @llvm.ve.vl.vseq.vl(i32 256)
  vr1 = _vel_vseq_vl(256);
}

void __attribute__((noinline))
test_vseq_vvl() {
  // CHECK-LABEL: @test_vseq_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vseq.vvl(<256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vseq_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvseqlo_vl() {
  // CHECK-LABEL: @test_pvseqlo_vl
  // CHECK: call <256 x double> @llvm.ve.vl.pvseqlo.vl(i32 256)
  vr1 = _vel_pvseqlo_vl(256);
}

void __attribute__((noinline))
test_pvseqlo_vvl() {
  // CHECK-LABEL: @test_pvseqlo_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvseqlo.vvl(<256 x double> %{{.*}}, i32 256)
  vr1 = _vel_pvseqlo_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvsequp_vl() {
  // CHECK-LABEL: @test_pvsequp_vl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsequp.vl(i32 256)
  vr1 = _vel_pvsequp_vl(256);
}

void __attribute__((noinline))
test_pvsequp_vvl() {
  // CHECK-LABEL: @test_pvsequp_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsequp.vvl(<256 x double> %{{.*}}, i32 256)
  vr1 = _vel_pvsequp_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvseq_vl() {
  // CHECK-LABEL: @test_pvseq_vl
  // CHECK: call <256 x double> @llvm.ve.vl.pvseq.vl(i32 256)
  vr1 = _vel_pvseq_vl(256);
}

void __attribute__((noinline))
test_pvseq_vvl() {
  // CHECK-LABEL: @test_pvseq_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvseq.vvl(<256 x double> %{{.*}}, i32 256)
  vr1 = _vel_pvseq_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsll_vvvl() {
  // CHECK-LABEL: @test_vsll_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsll_vvvvl() {
  // CHECK-LABEL: @test_vsll_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsll_vvsl() {
  // CHECK-LABEL: @test_vsll_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vsll_vvsvl() {
  // CHECK-LABEL: @test_vsll_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_pvsll_vvvl() {
  // CHECK-LABEL: @test_pvsll_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsll_vvvvl() {
  // CHECK-LABEL: @test_pvsll_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsll_vvsl() {
  // CHECK-LABEL: @test_pvsll_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_pvsll_vvsvl() {
  // CHECK-LABEL: @test_pvsll_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsrl_vvvl() {
  // CHECK-LABEL: @test_vsrl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsrl_vvvvl() {
  // CHECK-LABEL: @test_vsrl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsrl_vvsl() {
  // CHECK-LABEL: @test_vsrl_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vsrl_vvsvl() {
  // CHECK-LABEL: @test_vsrl_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_pvsrl_vvvl() {
  // CHECK-LABEL: @test_pvsrl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsrl_vvvvl() {
  // CHECK-LABEL: @test_pvsrl_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsrl_vvsl() {
  // CHECK-LABEL: @test_pvsrl_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_pvsrl_vvsvl() {
  // CHECK-LABEL: @test_pvsrl_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vslawsx_vvvl() {
  // CHECK-LABEL: @test_vslawsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vslawsx_vvvvl() {
  // CHECK-LABEL: @test_vslawsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vslawsx_vvsl() {
  // CHECK-LABEL: @test_vslawsx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vslawsx_vvsvl() {
  // CHECK-LABEL: @test_vslawsx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vslawzx_vvvl() {
  // CHECK-LABEL: @test_vslawzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vslawzx_vvvvl() {
  // CHECK-LABEL: @test_vslawzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vslawzx_vvsl() {
  // CHECK-LABEL: @test_vslawzx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vslawzx_vvsvl() {
  // CHECK-LABEL: @test_vslawzx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_pvsla_vvvl() {
  // CHECK-LABEL: @test_pvsla_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsla_vvvvl() {
  // CHECK-LABEL: @test_pvsla_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsla_vvsl() {
  // CHECK-LABEL: @test_pvsla_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_pvsla_vvsvl() {
  // CHECK-LABEL: @test_pvsla_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vslal_vvvl() {
  // CHECK-LABEL: @test_vslal_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vslal_vvvvl() {
  // CHECK-LABEL: @test_vslal_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vslal_vvsl() {
  // CHECK-LABEL: @test_vslal_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vslal_vvsvl() {
  // CHECK-LABEL: @test_vslal_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsrawsx_vvvl() {
  // CHECK-LABEL: @test_vsrawsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsrawsx_vvvvl() {
  // CHECK-LABEL: @test_vsrawsx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsrawsx_vvsl() {
  // CHECK-LABEL: @test_vsrawsx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vsrawsx_vvsvl() {
  // CHECK-LABEL: @test_vsrawsx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsrawzx_vvvl() {
  // CHECK-LABEL: @test_vsrawzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsrawzx_vvvvl() {
  // CHECK-LABEL: @test_vsrawzx_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsrawzx_vvsl() {
  // CHECK-LABEL: @test_vsrawzx_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvsl(<256 x double> %{{.*}}, i32 %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vsrawzx_vvsvl() {
  // CHECK-LABEL: @test_vsrawzx_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvsvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_pvsra_vvvl() {
  // CHECK-LABEL: @test_pvsra_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvsra_vvvvl() {
  // CHECK-LABEL: @test_pvsra_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvsra_vvsl() {
  // CHECK-LABEL: @test_pvsra_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_pvsra_vvsvl() {
  // CHECK-LABEL: @test_pvsra_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsral_vvvl() {
  // CHECK-LABEL: @test_vsral_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vsral_vvvvl() {
  // CHECK-LABEL: @test_vsral_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vsral_vvsl() {
  // CHECK-LABEL: @test_vsral_vvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvsl(<256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvsl(vr1, v2, 256);
}

void __attribute__((noinline))
test_vsral_vvsvl() {
  // CHECK-LABEL: @test_vsral_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvsvl(vr1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsfa_vvssl() {
  // CHECK-LABEL: @test_vsfa_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vsfa.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vsfa_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vsfa_vvssvl() {
  // CHECK-LABEL: @test_vsfa_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsfa.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsfa_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vfaddd_vvvl() {
  // CHECK-LABEL: @test_vfaddd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfaddd_vvvvl() {
  // CHECK-LABEL: @test_vfaddd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfaddd_vsvl() {
  // CHECK-LABEL: @test_vfaddd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfaddd_vsvvl() {
  // CHECK-LABEL: @test_vfaddd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfadds_vvvl() {
  // CHECK-LABEL: @test_vfadds_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfadds_vvvvl() {
  // CHECK-LABEL: @test_vfadds_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfadds_vsvl() {
  // CHECK-LABEL: @test_vfadds_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfadds_vsvvl() {
  // CHECK-LABEL: @test_vfadds_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfadd_vvvl() {
  // CHECK-LABEL: @test_pvfadd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfadd_vvvvl() {
  // CHECK-LABEL: @test_pvfadd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfadd_vsvl() {
  // CHECK-LABEL: @test_pvfadd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfadd_vsvvl() {
  // CHECK-LABEL: @test_pvfadd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfsubd_vvvl() {
  // CHECK-LABEL: @test_vfsubd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfsubd_vvvvl() {
  // CHECK-LABEL: @test_vfsubd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfsubd_vsvl() {
  // CHECK-LABEL: @test_vfsubd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfsubd_vsvvl() {
  // CHECK-LABEL: @test_vfsubd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfsubs_vvvl() {
  // CHECK-LABEL: @test_vfsubs_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfsubs_vvvvl() {
  // CHECK-LABEL: @test_vfsubs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfsubs_vsvl() {
  // CHECK-LABEL: @test_vfsubs_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfsubs_vsvvl() {
  // CHECK-LABEL: @test_vfsubs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfsub_vvvl() {
  // CHECK-LABEL: @test_pvfsub_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfsub_vvvvl() {
  // CHECK-LABEL: @test_pvfsub_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfsub_vsvl() {
  // CHECK-LABEL: @test_pvfsub_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfsub_vsvvl() {
  // CHECK-LABEL: @test_pvfsub_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmuld_vvvl() {
  // CHECK-LABEL: @test_vfmuld_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmuld_vvvvl() {
  // CHECK-LABEL: @test_vfmuld_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmuld_vsvl() {
  // CHECK-LABEL: @test_vfmuld_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfmuld_vsvvl() {
  // CHECK-LABEL: @test_vfmuld_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmuls_vvvl() {
  // CHECK-LABEL: @test_vfmuls_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmuls_vvvvl() {
  // CHECK-LABEL: @test_vfmuls_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmuls_vsvl() {
  // CHECK-LABEL: @test_vfmuls_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfmuls_vsvvl() {
  // CHECK-LABEL: @test_vfmuls_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmul_vvvl() {
  // CHECK-LABEL: @test_pvfmul_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmul_vvvvl() {
  // CHECK-LABEL: @test_pvfmul_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmul_vsvl() {
  // CHECK-LABEL: @test_pvfmul_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmul_vsvvl() {
  // CHECK-LABEL: @test_pvfmul_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfdivd_vvvl() {
  // CHECK-LABEL: @test_vfdivd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfdivd_vvvvl() {
  // CHECK-LABEL: @test_vfdivd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfdivd_vsvl() {
  // CHECK-LABEL: @test_vfdivd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfdivd_vsvvl() {
  // CHECK-LABEL: @test_vfdivd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfdivs_vvvl() {
  // CHECK-LABEL: @test_vfdivs_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfdivs_vvvvl() {
  // CHECK-LABEL: @test_vfdivs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfdivs_vsvl() {
  // CHECK-LABEL: @test_vfdivs_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfdivs_vsvvl() {
  // CHECK-LABEL: @test_vfdivs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfsqrtd_vvl() {
  // CHECK-LABEL: @test_vfsqrtd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsqrtd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsqrtd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsqrtd_vvvl() {
  // CHECK-LABEL: @test_vfsqrtd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsqrtd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsqrtd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfsqrts_vvl() {
  // CHECK-LABEL: @test_vfsqrts_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsqrts.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsqrts_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsqrts_vvvl() {
  // CHECK-LABEL: @test_vfsqrts_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsqrts.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsqrts_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfcmpd_vvvl() {
  // CHECK-LABEL: @test_vfcmpd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfcmpd_vvvvl() {
  // CHECK-LABEL: @test_vfcmpd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfcmpd_vsvl() {
  // CHECK-LABEL: @test_vfcmpd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfcmpd_vsvvl() {
  // CHECK-LABEL: @test_vfcmpd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfcmps_vvvl() {
  // CHECK-LABEL: @test_vfcmps_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfcmps_vvvvl() {
  // CHECK-LABEL: @test_vfcmps_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfcmps_vsvl() {
  // CHECK-LABEL: @test_vfcmps_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfcmps_vsvvl() {
  // CHECK-LABEL: @test_vfcmps_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfcmp_vvvl() {
  // CHECK-LABEL: @test_pvfcmp_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfcmp_vvvvl() {
  // CHECK-LABEL: @test_pvfcmp_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfcmp_vsvl() {
  // CHECK-LABEL: @test_pvfcmp_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfcmp_vsvvl() {
  // CHECK-LABEL: @test_pvfcmp_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxd_vvvl() {
  // CHECK-LABEL: @test_vfmaxd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmaxd_vvvvl() {
  // CHECK-LABEL: @test_vfmaxd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxd_vsvl() {
  // CHECK-LABEL: @test_vfmaxd_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfmaxd_vsvvl() {
  // CHECK-LABEL: @test_vfmaxd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxs_vvvl() {
  // CHECK-LABEL: @test_vfmaxs_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmaxs_vvvvl() {
  // CHECK-LABEL: @test_vfmaxs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxs_vsvl() {
  // CHECK-LABEL: @test_vfmaxs_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfmaxs_vsvvl() {
  // CHECK-LABEL: @test_vfmaxs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmax_vvvl() {
  // CHECK-LABEL: @test_pvfmax_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmax_vvvvl() {
  // CHECK-LABEL: @test_pvfmax_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmax_vsvl() {
  // CHECK-LABEL: @test_pvfmax_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmax_vsvvl() {
  // CHECK-LABEL: @test_pvfmax_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmind_vvvl() {
  // CHECK-LABEL: @test_vfmind_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmind_vvvvl() {
  // CHECK-LABEL: @test_vfmind_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmind_vsvl() {
  // CHECK-LABEL: @test_vfmind_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vsvl(double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vsvl(vd1, vr2, 256);
}

void __attribute__((noinline))
test_vfmind_vsvvl() {
  // CHECK-LABEL: @test_vfmind_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmins_vvvl() {
  // CHECK-LABEL: @test_vfmins_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfmins_vvvvl() {
  // CHECK-LABEL: @test_vfmins_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmins_vsvl() {
  // CHECK-LABEL: @test_vfmins_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vsvl(float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vsvl(vf1, vr2, 256);
}

void __attribute__((noinline))
test_vfmins_vsvvl() {
  // CHECK-LABEL: @test_vfmins_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmin_vvvl() {
  // CHECK-LABEL: @test_pvfmin_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmin_vvvvl() {
  // CHECK-LABEL: @test_pvfmin_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmin_vsvl() {
  // CHECK-LABEL: @test_pvfmin_vsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vsvl(i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vsvl(v1, vr2, 256);
}

void __attribute__((noinline))
test_pvfmin_vsvvl() {
  // CHECK-LABEL: @test_pvfmin_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmadd_vvvvl() {
  // CHECK-LABEL: @test_vfmadd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmadd_vvvvvl() {
  // CHECK-LABEL: @test_vfmadd_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmadd_vsvvl() {
  // CHECK-LABEL: @test_vfmadd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmadd_vsvvvl() {
  // CHECK-LABEL: @test_vfmadd_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vsvvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vsvvvl(vd1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmadd_vvsvl() {
  // CHECK-LABEL: @test_vfmadd_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvsvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvsvl(vr1, vd1, vr3, 256);
}

void __attribute__((noinline))
test_vfmadd_vvsvvl() {
  // CHECK-LABEL: @test_vfmadd_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvsvvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvsvvl(vr1, vd1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmads_vvvvl() {
  // CHECK-LABEL: @test_vfmads_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmads_vvvvvl() {
  // CHECK-LABEL: @test_vfmads_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmads_vsvvl() {
  // CHECK-LABEL: @test_vfmads_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmads_vsvvvl() {
  // CHECK-LABEL: @test_vfmads_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vsvvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vsvvvl(vf1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmads_vvsvl() {
  // CHECK-LABEL: @test_vfmads_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvsvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvsvl(vr1, vf1, vr3, 256);
}

void __attribute__((noinline))
test_vfmads_vvsvvl() {
  // CHECK-LABEL: @test_vfmads_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvsvvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvsvvl(vr1, vf1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmad_vvvvl() {
  // CHECK-LABEL: @test_pvfmad_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmad_vvvvvl() {
  // CHECK-LABEL: @test_pvfmad_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmad_vsvvl() {
  // CHECK-LABEL: @test_pvfmad_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmad_vsvvvl() {
  // CHECK-LABEL: @test_pvfmad_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vsvvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vsvvvl(v1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmad_vvsvl() {
  // CHECK-LABEL: @test_pvfmad_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvsvl(vr1, v1, vr3, 256);
}

void __attribute__((noinline))
test_pvfmad_vvsvvl() {
  // CHECK-LABEL: @test_pvfmad_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvsvvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvsvvl(vr1, v1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbd_vvvvl() {
  // CHECK-LABEL: @test_vfmsbd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbd_vvvvvl() {
  // CHECK-LABEL: @test_vfmsbd_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbd_vsvvl() {
  // CHECK-LABEL: @test_vfmsbd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbd_vsvvvl() {
  // CHECK-LABEL: @test_vfmsbd_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vsvvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vsvvvl(vd1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbd_vvsvl() {
  // CHECK-LABEL: @test_vfmsbd_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvsvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvsvl(vr1, vd1, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbd_vvsvvl() {
  // CHECK-LABEL: @test_vfmsbd_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvsvvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvsvvl(vr1, vd1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbs_vvvvl() {
  // CHECK-LABEL: @test_vfmsbs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbs_vvvvvl() {
  // CHECK-LABEL: @test_vfmsbs_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbs_vsvvl() {
  // CHECK-LABEL: @test_vfmsbs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbs_vsvvvl() {
  // CHECK-LABEL: @test_vfmsbs_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vsvvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vsvvvl(vf1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbs_vvsvl() {
  // CHECK-LABEL: @test_vfmsbs_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvsvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvsvl(vr1, vf1, vr3, 256);
}

void __attribute__((noinline))
test_vfmsbs_vvsvvl() {
  // CHECK-LABEL: @test_vfmsbs_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvsvvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvsvvl(vr1, vf1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmsb_vvvvl() {
  // CHECK-LABEL: @test_pvfmsb_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmsb_vvvvvl() {
  // CHECK-LABEL: @test_pvfmsb_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmsb_vsvvl() {
  // CHECK-LABEL: @test_pvfmsb_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfmsb_vsvvvl() {
  // CHECK-LABEL: @test_pvfmsb_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vsvvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vsvvvl(v1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfmsb_vvsvl() {
  // CHECK-LABEL: @test_pvfmsb_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvsvl(vr1, v1, vr3, 256);
}

void __attribute__((noinline))
test_pvfmsb_vvsvvl() {
  // CHECK-LABEL: @test_pvfmsb_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvsvvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvsvvl(vr1, v1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmadd_vvvvl() {
  // CHECK-LABEL: @test_vfnmadd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmadd_vvvvvl() {
  // CHECK-LABEL: @test_vfnmadd_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmadd_vsvvl() {
  // CHECK-LABEL: @test_vfnmadd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmadd_vsvvvl() {
  // CHECK-LABEL: @test_vfnmadd_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vsvvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vsvvvl(vd1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmadd_vvsvl() {
  // CHECK-LABEL: @test_vfnmadd_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvsvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvsvl(vr1, vd1, vr3, 256);
}

void __attribute__((noinline))
test_vfnmadd_vvsvvl() {
  // CHECK-LABEL: @test_vfnmadd_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvsvvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvsvvl(vr1, vd1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmads_vvvvl() {
  // CHECK-LABEL: @test_vfnmads_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmads_vvvvvl() {
  // CHECK-LABEL: @test_vfnmads_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmads_vsvvl() {
  // CHECK-LABEL: @test_vfnmads_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmads_vsvvvl() {
  // CHECK-LABEL: @test_vfnmads_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vsvvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vsvvvl(vf1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmads_vvsvl() {
  // CHECK-LABEL: @test_vfnmads_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvsvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvsvl(vr1, vf1, vr3, 256);
}

void __attribute__((noinline))
test_vfnmads_vvsvvl() {
  // CHECK-LABEL: @test_vfnmads_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvsvvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvsvvl(vr1, vf1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmad_vvvvl() {
  // CHECK-LABEL: @test_pvfnmad_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmad_vvvvvl() {
  // CHECK-LABEL: @test_pvfnmad_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmad_vsvvl() {
  // CHECK-LABEL: @test_pvfnmad_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmad_vsvvvl() {
  // CHECK-LABEL: @test_pvfnmad_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vsvvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vsvvvl(v1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmad_vvsvl() {
  // CHECK-LABEL: @test_pvfnmad_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvsvl(vr1, v1, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmad_vvsvvl() {
  // CHECK-LABEL: @test_pvfnmad_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvsvvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvsvvl(vr1, v1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vvvvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vvvvvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vsvvl() {
  // CHECK-LABEL: @test_vfnmsbd_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vsvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vsvvl(vd1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vsvvvl() {
  // CHECK-LABEL: @test_vfnmsbd_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vsvvvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vsvvvl(vd1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vvsvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvsvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvsvl(vr1, vd1, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vvsvvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvsvvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvsvvl(vr1, vd1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vvvvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vvvvvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vsvvl() {
  // CHECK-LABEL: @test_vfnmsbs_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vsvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vsvvl(vf1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vsvvvl() {
  // CHECK-LABEL: @test_vfnmsbs_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vsvvvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vsvvvl(vf1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vvsvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvsvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvsvl(vr1, vf1, vr3, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vvsvvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvsvvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvsvvl(vr1, vf1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vvvvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvvvl(vr1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vvvvvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvvvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvvvvl(vr1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vsvvl() {
  // CHECK-LABEL: @test_pvfnmsb_vsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vsvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vsvvl(v1, vr2, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vsvvvl() {
  // CHECK-LABEL: @test_pvfnmsb_vsvvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vsvvvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vsvvvl(v1, vr2, vr3, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vvsvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvsvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvsvl(vr1, v1, vr3, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vvsvvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvsvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvsvvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvsvvl(vr1, v1, vr3, vr4, 256);
}

void __attribute__((noinline))
test_vrcpd_vvl() {
  // CHECK-LABEL: @test_vrcpd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrcpd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrcpd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrcpd_vvvl() {
  // CHECK-LABEL: @test_vrcpd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrcpd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrcpd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrcps_vvl() {
  // CHECK-LABEL: @test_vrcps_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrcps.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrcps_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrcps_vvvl() {
  // CHECK-LABEL: @test_vrcps_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrcps.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrcps_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvrcp_vvl() {
  // CHECK-LABEL: @test_pvrcp_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrcp.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvrcp_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvrcp_vvvl() {
  // CHECK-LABEL: @test_pvrcp_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrcp.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvrcp_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrsqrtd_vvl() {
  // CHECK-LABEL: @test_vrsqrtd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrsqrtd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrsqrtd_vvvl() {
  // CHECK-LABEL: @test_vrsqrtd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrsqrtd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrsqrts_vvl() {
  // CHECK-LABEL: @test_vrsqrts_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrts.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrsqrts_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrsqrts_vvvl() {
  // CHECK-LABEL: @test_vrsqrts_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrts.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrsqrts_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvrsqrt_vvl() {
  // CHECK-LABEL: @test_pvrsqrt_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrsqrt.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvrsqrt_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvrsqrt_vvvl() {
  // CHECK-LABEL: @test_pvrsqrt_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrsqrt.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvrsqrt_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrsqrtdnex_vvl() {
  // CHECK-LABEL: @test_vrsqrtdnex_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtdnex.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrsqrtdnex_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrsqrtdnex_vvvl() {
  // CHECK-LABEL: @test_vrsqrtdnex_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtdnex.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrsqrtdnex_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrsqrtsnex_vvl() {
  // CHECK-LABEL: @test_vrsqrtsnex_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtsnex.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vrsqrtsnex_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrsqrtsnex_vvvl() {
  // CHECK-LABEL: @test_vrsqrtsnex_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrsqrtsnex.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrsqrtsnex_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvrsqrtnex_vvl() {
  // CHECK-LABEL: @test_pvrsqrtnex_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrsqrtnex.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvrsqrtnex_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvrsqrtnex_vvvl() {
  // CHECK-LABEL: @test_pvrsqrtnex_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvrsqrtnex.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvrsqrtnex_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwdsx_vvl() {
  // CHECK-LABEL: @test_vcvtwdsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwdsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwdsx_vvvl() {
  // CHECK-LABEL: @test_vcvtwdsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwdsxrz_vvl() {
  // CHECK-LABEL: @test_vcvtwdsxrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsxrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwdsxrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwdsxrz_vvvl() {
  // CHECK-LABEL: @test_vcvtwdsxrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsxrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdsxrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwdzx_vvl() {
  // CHECK-LABEL: @test_vcvtwdzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwdzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwdzx_vvvl() {
  // CHECK-LABEL: @test_vcvtwdzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwdzxrz_vvl() {
  // CHECK-LABEL: @test_vcvtwdzxrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzxrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwdzxrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwdzxrz_vvvl() {
  // CHECK-LABEL: @test_vcvtwdzxrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzxrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdzxrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwssx_vvl() {
  // CHECK-LABEL: @test_vcvtwssx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwssx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwssx_vvvl() {
  // CHECK-LABEL: @test_vcvtwssx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwssx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwssxrz_vvl() {
  // CHECK-LABEL: @test_vcvtwssxrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssxrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwssxrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwssxrz_vvvl() {
  // CHECK-LABEL: @test_vcvtwssxrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssxrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwssxrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwszx_vvl() {
  // CHECK-LABEL: @test_vcvtwszx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwszx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwszx_vvvl() {
  // CHECK-LABEL: @test_vcvtwszx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwszx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtwszxrz_vvl() {
  // CHECK-LABEL: @test_vcvtwszxrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszxrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtwszxrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtwszxrz_vvvl() {
  // CHECK-LABEL: @test_vcvtwszxrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszxrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwszxrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvcvtws_vvl() {
  // CHECK-LABEL: @test_pvcvtws_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtws.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvcvtws_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvcvtws_vvvl() {
  // CHECK-LABEL: @test_pvcvtws_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtws.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcvtws_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvcvtwsrz_vvl() {
  // CHECK-LABEL: @test_pvcvtwsrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtwsrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvcvtwsrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvcvtwsrz_vvvl() {
  // CHECK-LABEL: @test_pvcvtwsrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtwsrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcvtwsrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtld_vvl() {
  // CHECK-LABEL: @test_vcvtld_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtld.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtld_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtld_vvvl() {
  // CHECK-LABEL: @test_vcvtld_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtld.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtld_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtldrz_vvl() {
  // CHECK-LABEL: @test_vcvtldrz_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtldrz.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtldrz_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtldrz_vvvl() {
  // CHECK-LABEL: @test_vcvtldrz_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtldrz.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtldrz_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtdw_vvl() {
  // CHECK-LABEL: @test_vcvtdw_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtdw.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtdw_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtdw_vvvl() {
  // CHECK-LABEL: @test_vcvtdw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtdw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtdw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtsw_vvl() {
  // CHECK-LABEL: @test_vcvtsw_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtsw.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtsw_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtsw_vvvl() {
  // CHECK-LABEL: @test_vcvtsw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtsw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtsw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_pvcvtsw_vvl() {
  // CHECK-LABEL: @test_pvcvtsw_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtsw.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_pvcvtsw_vvl(vr1, 256);
}

void __attribute__((noinline))
test_pvcvtsw_vvvl() {
  // CHECK-LABEL: @test_pvcvtsw_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtsw.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcvtsw_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtdl_vvl() {
  // CHECK-LABEL: @test_vcvtdl_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtdl.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtdl_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtdl_vvvl() {
  // CHECK-LABEL: @test_vcvtdl_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtdl.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtdl_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtds_vvl() {
  // CHECK-LABEL: @test_vcvtds_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtds.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtds_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtds_vvvl() {
  // CHECK-LABEL: @test_vcvtds_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtds.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtds_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vcvtsd_vvl() {
  // CHECK-LABEL: @test_vcvtsd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtsd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vcvtsd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vcvtsd_vvvl() {
  // CHECK-LABEL: @test_vcvtsd_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtsd.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtsd_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vshf_vvvsl() {
  // CHECK-LABEL: @test_vshf_vvvsl
  // CHECK: call <256 x double> @llvm.ve.vl.vshf.vvvsl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vshf_vvvsl(vr1, vr2, v1, 256);
}

void __attribute__((noinline))
test_vshf_vvvsvl() {
  // CHECK-LABEL: @test_vshf_vvvsvl
  // CHECK: call <256 x double> @llvm.ve.vl.vshf.vvvsvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vshf_vvvsvl(vr1, vr2, v1, vr3, 256);
}

void __attribute__((noinline))
test_vsumwsx_vvl() {
  // CHECK-LABEL: @test_vsumwsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsumwsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsumwzx_vvl() {
  // CHECK-LABEL: @test_vsumwzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsumwzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsuml_vvl() {
  // CHECK-LABEL: @test_vsuml_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsuml.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsuml_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsumd_vvl() {
  // CHECK-LABEL: @test_vfsumd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsumd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsumd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsums_vvl() {
  // CHECK-LABEL: @test_vfsums_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsums.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsums_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxswfstsx_vvl() {
  // CHECK-LABEL: @test_vrmaxswfstsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswfstsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswfstsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxswfstsx_vvvl() {
  // CHECK-LABEL: @test_vrmaxswfstsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswfstsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswfstsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrmaxswlstsx_vvl() {
  // CHECK-LABEL: @test_vrmaxswlstsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswlstsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswlstsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxswlstsx_vvvl() {
  // CHECK-LABEL: @test_vrmaxswlstsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswlstsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswlstsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrmaxswfstzx_vvl() {
  // CHECK-LABEL: @test_vrmaxswfstzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswfstzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswfstzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxswfstzx_vvvl() {
  // CHECK-LABEL: @test_vrmaxswfstzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswfstzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswfstzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrmaxswlstzx_vvl() {
  // CHECK-LABEL: @test_vrmaxswlstzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswlstzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswlstzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxswlstzx_vvvl() {
  // CHECK-LABEL: @test_vrmaxswlstzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxswlstzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxswlstzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminswfstsx_vvl() {
  // CHECK-LABEL: @test_vrminswfstsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswfstsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswfstsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminswfstsx_vvvl() {
  // CHECK-LABEL: @test_vrminswfstsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswfstsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswfstsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminswlstsx_vvl() {
  // CHECK-LABEL: @test_vrminswlstsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswlstsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswlstsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminswlstsx_vvvl() {
  // CHECK-LABEL: @test_vrminswlstsx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswlstsx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswlstsx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminswfstzx_vvl() {
  // CHECK-LABEL: @test_vrminswfstzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswfstzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswfstzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminswfstzx_vvvl() {
  // CHECK-LABEL: @test_vrminswfstzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswfstzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswfstzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminswlstzx_vvl() {
  // CHECK-LABEL: @test_vrminswlstzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswlstzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswlstzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminswlstzx_vvvl() {
  // CHECK-LABEL: @test_vrminswlstzx_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminswlstzx.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminswlstzx_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrmaxslfst_vvl() {
  // CHECK-LABEL: @test_vrmaxslfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxslfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxslfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxslfst_vvvl() {
  // CHECK-LABEL: @test_vrmaxslfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxslfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxslfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrmaxsllst_vvl() {
  // CHECK-LABEL: @test_vrmaxsllst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxsllst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxsllst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrmaxsllst_vvvl() {
  // CHECK-LABEL: @test_vrmaxsllst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrmaxsllst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrmaxsllst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminslfst_vvl() {
  // CHECK-LABEL: @test_vrminslfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminslfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminslfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminslfst_vvvl() {
  // CHECK-LABEL: @test_vrminslfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminslfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminslfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrminsllst_vvl() {
  // CHECK-LABEL: @test_vrminsllst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminsllst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminsllst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrminsllst_vvvl() {
  // CHECK-LABEL: @test_vrminsllst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrminsllst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrminsllst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmaxdfst_vvl() {
  // CHECK-LABEL: @test_vfrmaxdfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxdfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxdfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmaxdfst_vvvl() {
  // CHECK-LABEL: @test_vfrmaxdfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxdfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxdfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmaxdlst_vvl() {
  // CHECK-LABEL: @test_vfrmaxdlst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxdlst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxdlst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmaxdlst_vvvl() {
  // CHECK-LABEL: @test_vfrmaxdlst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxdlst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxdlst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmaxsfst_vvl() {
  // CHECK-LABEL: @test_vfrmaxsfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxsfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxsfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmaxsfst_vvvl() {
  // CHECK-LABEL: @test_vfrmaxsfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxsfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxsfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmaxslst_vvl() {
  // CHECK-LABEL: @test_vfrmaxslst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxslst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxslst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmaxslst_vvvl() {
  // CHECK-LABEL: @test_vfrmaxslst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmaxslst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmaxslst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmindfst_vvl() {
  // CHECK-LABEL: @test_vfrmindfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmindfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmindfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmindfst_vvvl() {
  // CHECK-LABEL: @test_vfrmindfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmindfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmindfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrmindlst_vvl() {
  // CHECK-LABEL: @test_vfrmindlst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmindlst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmindlst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrmindlst_vvvl() {
  // CHECK-LABEL: @test_vfrmindlst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrmindlst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrmindlst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrminsfst_vvl() {
  // CHECK-LABEL: @test_vfrminsfst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrminsfst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrminsfst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrminsfst_vvvl() {
  // CHECK-LABEL: @test_vfrminsfst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrminsfst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrminsfst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vfrminslst_vvl() {
  // CHECK-LABEL: @test_vfrminslst_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrminslst.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrminslst_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfrminslst_vvvl() {
  // CHECK-LABEL: @test_vfrminslst_vvvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfrminslst.vvvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfrminslst_vvvl(vr1, vr2, 256);
}

void __attribute__((noinline))
test_vrand_vvl() {
  // CHECK-LABEL: @test_vrand_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrand.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrand_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vror_vvl() {
  // CHECK-LABEL: @test_vror_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vror.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vror_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrxor_vvl() {
  // CHECK-LABEL: @test_vrxor_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrxor.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrxor_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vgt_vvssl() {
  // CHECK-LABEL: @test_vgt_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgt.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgt_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgt_vvssvl() {
  // CHECK-LABEL: @test_vgt_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgt.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgt_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtnc_vvssl() {
  // CHECK-LABEL: @test_vgtnc_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtnc.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtnc_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtnc_vvssvl() {
  // CHECK-LABEL: @test_vgtnc_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtnc.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtnc_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtu_vvssl() {
  // CHECK-LABEL: @test_vgtu_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtu.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtu_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtu_vvssvl() {
  // CHECK-LABEL: @test_vgtu_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtu.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtu_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtunc_vvssl() {
  // CHECK-LABEL: @test_vgtunc_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtunc.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtunc_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtunc_vvssvl() {
  // CHECK-LABEL: @test_vgtunc_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtunc.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtunc_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtlsx_vvssl() {
  // CHECK-LABEL: @test_vgtlsx_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsx.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtlsx_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtlsx_vvssvl() {
  // CHECK-LABEL: @test_vgtlsx_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsx.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsx_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtlsxnc_vvssl() {
  // CHECK-LABEL: @test_vgtlsxnc_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsxnc.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtlsxnc_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtlsxnc_vvssvl() {
  // CHECK-LABEL: @test_vgtlsxnc_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsxnc.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsxnc_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtlzx_vvssl() {
  // CHECK-LABEL: @test_vgtlzx_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzx.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtlzx_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtlzx_vvssvl() {
  // CHECK-LABEL: @test_vgtlzx_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzx.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzx_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vgtlzxnc_vvssl() {
  // CHECK-LABEL: @test_vgtlzxnc_vvssl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzxnc.vvssl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  vr3 = _vel_vgtlzxnc_vvssl(vr1, v1, v2, 256);
}

void __attribute__((noinline))
test_vgtlzxnc_vvssvl() {
  // CHECK-LABEL: @test_vgtlzxnc_vvssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzxnc.vvssvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzxnc_vvssvl(vr1, v1, v2, vr3, 256);
}

void __attribute__((noinline))
test_vsc_vvssl() {
  // CHECK-LABEL: @test_vsc_vvssl
  // CHECK: call void @llvm.ve.vl.vsc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscnc_vvssl() {
  // CHECK-LABEL: @test_vscnc_vvssl
  // CHECK: call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscnc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscot_vvssl() {
  // CHECK-LABEL: @test_vscot_vvssl
  // CHECK: call void @llvm.ve.vl.vscot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscncot_vvssl() {
  // CHECK-LABEL: @test_vscncot_vvssl
  // CHECK: call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscu_vvssl() {
  // CHECK-LABEL: @test_vscu_vvssl
  // CHECK: call void @llvm.ve.vl.vscu.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscu_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscunc_vvssl() {
  // CHECK-LABEL: @test_vscunc_vvssl
  // CHECK: call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscunc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscuot_vvssl() {
  // CHECK-LABEL: @test_vscuot_vvssl
  // CHECK: call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscuot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscuncot_vvssl() {
  // CHECK-LABEL: @test_vscuncot_vvssl
  // CHECK: call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscuncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscl_vvssl() {
  // CHECK-LABEL: @test_vscl_vvssl
  // CHECK: call void @llvm.ve.vl.vscl.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscl_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclnc_vvssl() {
  // CHECK-LABEL: @test_vsclnc_vvssl
  // CHECK: call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclnc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclot_vvssl() {
  // CHECK-LABEL: @test_vsclot_vvssl
  // CHECK: call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclncot_vvssl() {
  // CHECK-LABEL: @test_vsclncot_vvssl
  // CHECK: call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_lcr_sss() {
  // CHECK-LABEL: @test_lcr_sss
  // CHECK: call i64 @llvm.ve.vl.lcr.sss(i64 %{{.*}}, i64 %{{.*}})
  v3 = _vel_lcr_sss(v1, v2);
}

void __attribute__((noinline))
test_scr_sss() {
  // CHECK-LABEL: @test_scr_sss
  // CHECK: call void @llvm.ve.vl.scr.sss(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  _vel_scr_sss(v1, v2, v3);
}

void __attribute__((noinline))
test_tscr_ssss() {
  // CHECK-LABEL: @test_tscr_ssss
  // CHECK: call i64 @llvm.ve.vl.tscr.ssss(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  v3 = _vel_tscr_ssss(v1, v2, v3);
}

void __attribute__((noinline))
test_fidcr_sss() {
  // CHECK-LABEL: @test_fidcr_sss
  // CHECK: call i64 @llvm.ve.vl.fidcr.sss(i64 %{{.*}}, i32 0)
  v2 = _vel_fidcr_sss(v1, 0);
}

void __attribute__((noinline))
test_fencei() {
  // CHECK-LABEL: @test_fencei
  // CHECK: call void @llvm.ve.vl.fencei()
  _vel_fencei();
}

void __attribute__((noinline))
test_fencem_s() {
  // CHECK-LABEL: @test_fencem_s
  // CHECK: call void @llvm.ve.vl.fencem.s(i32 0)
  _vel_fencem_s(0);
}

void __attribute__((noinline))
test_fencec_s() {
  // CHECK-LABEL: @test_fencec_s
  // CHECK: call void @llvm.ve.vl.fencec.s(i32 0)
  _vel_fencec_s(0);
}

void __attribute__((noinline))
test_svob() {
  // CHECK-LABEL: @test_svob
  // CHECK: call void @llvm.ve.vl.svob()
  _vel_svob();
}
