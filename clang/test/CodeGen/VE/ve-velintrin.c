// REQUIRES: ve-registered-target

// RUN: %clang_cc1 -S -emit-llvm -triple ve-unknown-linux-gnu \
// RUN:   -ffreestanding %s -o - | FileCheck %s

#include <velintrin.h>

long v1, v2, v3;
double vd1;
float vf1;
__vr vr1, vr2, vr3, vr4;
__vm256 vm1, vm2, vm3;
__vm512 vm1_512, vm2_512, vm3_512;

void __attribute__((noinline))
test_vld_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vld_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldu_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldunc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldunc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldunc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldunc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldunc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldunc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldunc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldunc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlsx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsx.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldlsx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlsx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsx.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlsx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlsxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsxnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldlsxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlsxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlsxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlsxnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlsxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlzx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzx.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldlzx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlzx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzx.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlzx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldlzxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzxnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldlzxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldlzxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldlzxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldlzxnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldlzxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vld2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2d_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2d.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vld2d_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld2d_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2d_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2d.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld2d_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vld2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2dnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2dnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vld2dnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vld2dnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vld2dnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vld2dnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vld2dnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2d_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2d.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldu2d_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu2d_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2d_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2d.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu2d_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldu2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2dnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2dnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldu2dnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldu2dnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldu2dnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldu2dnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldu2dnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dsx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsx.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dsx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsx.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dsxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsxnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dsxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dsxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dsxnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dsxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dzx_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzx_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzx.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzx_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dzx_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzx_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzx.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzx_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vldl2dzxnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzxnc_vssl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzxnc.vssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzxnc_vssl(idx, p, 256);
}

void __attribute__((noinline))
test_vldl2dzxnc_vssvl(char* p, long idx) {
  // CHECK-LABEL: @test_vldl2dzxnc_vssvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldl2dzxnc.vssvl(i64 %{{.*}}, ptr %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vldl2dzxnc_vssvl(idx, p, vr1, 256);
}

void __attribute__((noinline))
test_vst_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst_vssl
  // CHECK: call void @llvm.ve.vl.vst.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vst_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vst_vssml
  // CHECK: call void @llvm.ve.vl.vst.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vst_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstnc_vssl
  // CHECK: call void @llvm.ve.vl.vstnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstnc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstnc_vssml
  // CHECK: call void @llvm.ve.vl.vstnc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstnc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstot_vssl
  // CHECK: call void @llvm.ve.vl.vstot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstot_vssml
  // CHECK: call void @llvm.ve.vl.vstot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstncot_vssl
  // CHECK: call void @llvm.ve.vl.vstncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstncot_vssml
  // CHECK: call void @llvm.ve.vl.vstncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstu_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu_vssl
  // CHECK: call void @llvm.ve.vl.vstu.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstu_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstu_vssml
  // CHECK: call void @llvm.ve.vl.vstu.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstu_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstunc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstunc_vssl
  // CHECK: call void @llvm.ve.vl.vstunc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstunc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstunc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstunc_vssml
  // CHECK: call void @llvm.ve.vl.vstunc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstunc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstuot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstuot_vssl
  // CHECK: call void @llvm.ve.vl.vstuot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstuot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstuot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstuot_vssml
  // CHECK: call void @llvm.ve.vl.vstuot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstuot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstuncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstuncot_vssl
  // CHECK: call void @llvm.ve.vl.vstuncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstuncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstuncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstuncot_vssml
  // CHECK: call void @llvm.ve.vl.vstuncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstuncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstl_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl_vssl
  // CHECK: call void @llvm.ve.vl.vstl.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstl_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstl_vssml
  // CHECK: call void @llvm.ve.vl.vstl.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstl_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstlnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlnc_vssl
  // CHECK: call void @llvm.ve.vl.vstlnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstlnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlnc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstlnc_vssml
  // CHECK: call void @llvm.ve.vl.vstlnc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstlnc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstlot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlot_vssl
  // CHECK: call void @llvm.ve.vl.vstlot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstlot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstlot_vssml
  // CHECK: call void @llvm.ve.vl.vstlot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstlot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstlncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstlncot_vssl
  // CHECK: call void @llvm.ve.vl.vstlncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstlncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstlncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstlncot_vssml
  // CHECK: call void @llvm.ve.vl.vstlncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstlncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vst2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2d_vssl
  // CHECK: call void @llvm.ve.vl.vst2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vst2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2d_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vst2d_vssml
  // CHECK: call void @llvm.ve.vl.vst2d.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vst2d_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vst2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vst2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vst2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dnc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dnc_vssml
  // CHECK: call void @llvm.ve.vl.vst2dnc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vst2dnc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vst2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dot_vssl
  // CHECK: call void @llvm.ve.vl.vst2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vst2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dot_vssml
  // CHECK: call void @llvm.ve.vl.vst2dot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vst2dot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vst2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vst2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vst2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vst2dncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vst2dncot_vssml
  // CHECK: call void @llvm.ve.vl.vst2dncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vst2dncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstu2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2d_vssl
  // CHECK: call void @llvm.ve.vl.vstu2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstu2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2d_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2d_vssml
  // CHECK: call void @llvm.ve.vl.vstu2d.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstu2d_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstu2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstu2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dnc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dnc_vssml
  // CHECK: call void @llvm.ve.vl.vstu2dnc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstu2dnc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstu2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dot_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstu2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dot_vssml
  // CHECK: call void @llvm.ve.vl.vstu2dot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstu2dot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstu2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vstu2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstu2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstu2dncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstu2dncot_vssml
  // CHECK: call void @llvm.ve.vl.vstu2dncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstu2dncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstl2d_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2d_vssl
  // CHECK: call void @llvm.ve.vl.vstl2d.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstl2d_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2d_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2d_vssml
  // CHECK: call void @llvm.ve.vl.vstl2d.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstl2d_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstl2dnc_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dnc_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dnc.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstl2dnc_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dnc_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dnc_vssml
  // CHECK: call void @llvm.ve.vl.vstl2dnc.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstl2dnc_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstl2dot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dot_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstl2dot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dot_vssml
  // CHECK: call void @llvm.ve.vl.vstl2dot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstl2dot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_vstl2dncot_vssl(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dncot_vssl
  // CHECK: call void @llvm.ve.vl.vstl2dncot.vssl(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_vstl2dncot_vssl(vr1, idx, p, 256);
}

void __attribute__((noinline))
test_vstl2dncot_vssml(char* p, long idx) {
  // CHECK-LABEL: @test_vstl2dncot_vssml
  // CHECK: call void @llvm.ve.vl.vstl2dncot.vssml(<256 x double> %{{.*}}, i64 %{{.*}}, ptr %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vstl2dncot_vssml(vr1, idx, p, vm1, 256);
}

void __attribute__((noinline))
test_pfchv_ssl(char* p, long idx) {
  // CHECK-LABEL: @test_pfchv_ssl
  // CHECK: call void @llvm.ve.vl.pfchv.ssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
  _vel_pfchv_ssl(idx, p, 256);
}

void __attribute__((noinline))
test_pfchvnc_ssl(char* p, long idx) {
  // CHECK-LABEL: @test_pfchvnc_ssl
  // CHECK: call void @llvm.ve.vl.pfchvnc.ssl(i64 %{{.*}}, ptr %{{.*}}, i32 256)
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
test_lvm_mmss(unsigned long sy, unsigned long sz) {
  // CHECK-LABEL: @test_lvm_mmss
  // CHECK: call <256 x i1> @llvm.ve.vl.lvm.mmss(<256 x i1> %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  vm1 = _vel_lvm_mmss(vm2, sy, sz);
}

void __attribute__((noinline))
test_lvm_MMss(unsigned long sy, unsigned long sz) {
  // CHECK-LABEL: @test_lvm_MMss
  // CHECK: call <512 x i1> @llvm.ve.vl.lvm.MMss(<512 x i1> %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
  vm1_512 = _vel_lvm_MMss(vm2_512, sy, sz);
}

void __attribute__((noinline))
test_svm_sms(unsigned long sy) {
  // CHECK-LABEL: @test_svm_sms
  // CHECK: call i64 @llvm.ve.vl.svm.sms(<256 x i1> %{{.*}}, i64 %{{.*}})
  v1 = _vel_svm_sms(vm2, sy);
}

void __attribute__((noinline))
test_svm_sMs(unsigned long sy) {
  // CHECK-LABEL: @test_svm_sMs
  // CHECK: call i64 @llvm.ve.vl.svm.sMs(<512 x i1> %{{.*}}, i64 %{{.*}})
  v1 = _vel_svm_sMs(vm2_512, sy);
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
test_vbrdd_vsmvl() {
  // CHECK-LABEL: @test_vbrdd_vsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdd.vsmvl(double %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdd_vsmvl(vd1, vm1, vr1, 256);
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
test_vbrdl_vsmvl() {
  // CHECK-LABEL: @test_vbrdl_vsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdl.vsmvl(i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdl_vsmvl(v1, vm1, vr1, 256);
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
test_vbrds_vsmvl() {
  // CHECK-LABEL: @test_vbrds_vsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrds.vsmvl(float %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrds_vsmvl(vf1, vm1, vr1, 256);
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
test_vbrdw_vsmvl() {
  // CHECK-LABEL: @test_vbrdw_vsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrdw.vsmvl(i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vbrdw_vsmvl(v1, vm1, vr1, 256);
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
test_pvbrd_vsmvl() {
  // CHECK-LABEL: @test_pvbrd_vsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrd.vsMvl(i64 %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_pvbrd_vsMvl(v1, vm1_512, vr1, 256);
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
test_vmv_vsvmvl() {
  // CHECK-LABEL: @test_vmv_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmv.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr1 = _vel_vmv_vsvmvl(v1, vr1, vm1, vr2, 256);
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
test_vaddul_vvvmvl() {
  // CHECK-LABEL: @test_vaddul_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vaddul_vsvmvl() {
  // CHECK-LABEL: @test_vaddul_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddul.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddul_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vadduw_vvvmvl() {
  // CHECK-LABEL: @test_vadduw_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vadduw_vsvmvl() {
  // CHECK-LABEL: @test_vadduw_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vadduw.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vadduw_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvaddu_vvvMvl() {
  // CHECK-LABEL: @test_pvaddu_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvaddu_vsvMvl() {
  // CHECK-LABEL: @test_pvaddu_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvaddu.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvaddu_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vaddswsx_vvvmvl() {
  // CHECK-LABEL: @test_vaddswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vaddswsx_vsvmvl() {
  // CHECK-LABEL: @test_vaddswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vaddswzx_vvvmvl() {
  // CHECK-LABEL: @test_vaddswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vaddswzx_vsvmvl() {
  // CHECK-LABEL: @test_vaddswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvadds_vvvMvl() {
  // CHECK-LABEL: @test_pvadds_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvadds_vsvMvl() {
  // CHECK-LABEL: @test_pvadds_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvadds.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvadds_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vaddsl_vvvmvl() {
  // CHECK-LABEL: @test_vaddsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vaddsl_vsvmvl() {
  // CHECK-LABEL: @test_vaddsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vaddsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vaddsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vsubul_vvvmvl() {
  // CHECK-LABEL: @test_vsubul_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsubul_vsvmvl() {
  // CHECK-LABEL: @test_vsubul_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubul.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubul_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vsubuw_vvvmvl() {
  // CHECK-LABEL: @test_vsubuw_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsubuw_vsvmvl() {
  // CHECK-LABEL: @test_vsubuw_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubuw.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubuw_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvsubu_vvvMvl() {
  // CHECK-LABEL: @test_pvsubu_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsubu_vsvMvl() {
  // CHECK-LABEL: @test_pvsubu_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubu.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubu_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vsubswsx_vvvmvl() {
  // CHECK-LABEL: @test_vsubswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsubswsx_vsvmvl() {
  // CHECK-LABEL: @test_vsubswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vsubswzx_vvvmvl() {
  // CHECK-LABEL: @test_vsubswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsubswzx_vsvmvl() {
  // CHECK-LABEL: @test_vsubswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvsubs_vvvMvl() {
  // CHECK-LABEL: @test_pvsubs_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsubs_vsvMvl() {
  // CHECK-LABEL: @test_pvsubs_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsubs.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsubs_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vsubsl_vvvmvl() {
  // CHECK-LABEL: @test_vsubsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsubsl_vsvmvl() {
  // CHECK-LABEL: @test_vsubsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsubsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsubsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmulul_vvvmvl() {
  // CHECK-LABEL: @test_vmulul_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmulul_vsvmvl() {
  // CHECK-LABEL: @test_vmulul_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulul.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulul_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmuluw_vvvmvl() {
  // CHECK-LABEL: @test_vmuluw_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmuluw_vsvmvl() {
  // CHECK-LABEL: @test_vmuluw_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmuluw.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmuluw_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmulswsx_vvvmvl() {
  // CHECK-LABEL: @test_vmulswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmulswsx_vsvmvl() {
  // CHECK-LABEL: @test_vmulswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmulswzx_vvvmvl() {
  // CHECK-LABEL: @test_vmulswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmulswzx_vsvmvl() {
  // CHECK-LABEL: @test_vmulswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmulsl_vvvmvl() {
  // CHECK-LABEL: @test_vmulsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmulsl_vsvmvl() {
  // CHECK-LABEL: @test_vmulsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmulsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmulsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivul_vvvmvl() {
  // CHECK-LABEL: @test_vdivul_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vdivul_vsvmvl() {
  // CHECK-LABEL: @test_vdivul_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivuw_vvvmvl() {
  // CHECK-LABEL: @test_vdivuw_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vdivuw_vsvmvl() {
  // CHECK-LABEL: @test_vdivuw_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivul_vvsmvl() {
  // CHECK-LABEL: @test_vdivul_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivul.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivul_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vdivuw_vvsmvl() {
  // CHECK-LABEL: @test_vdivuw_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivuw.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivuw_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vdivswsx_vvvmvl() {
  // CHECK-LABEL: @test_vdivswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vdivswsx_vsvmvl() {
  // CHECK-LABEL: @test_vdivswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivswzx_vvvmvl() {
  // CHECK-LABEL: @test_vdivswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vdivswzx_vsvmvl() {
  // CHECK-LABEL: @test_vdivswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivswsx_vvsmvl() {
  // CHECK-LABEL: @test_vdivswsx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswsx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswsx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vdivswzx_vvsmvl() {
  // CHECK-LABEL: @test_vdivswzx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivswzx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivswzx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vdivsl_vvvmvl() {
  // CHECK-LABEL: @test_vdivsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vdivsl_vsvmvl() {
  // CHECK-LABEL: @test_vdivsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vdivsl_vvsmvl() {
  // CHECK-LABEL: @test_vdivsl_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vdivsl.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vdivsl_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vcmpul_vvvmvl() {
  // CHECK-LABEL: @test_vcmpul_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vcmpul_vsvmvl() {
  // CHECK-LABEL: @test_vcmpul_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpul.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpul_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vcmpuw_vvvmvl() {
  // CHECK-LABEL: @test_vcmpuw_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vcmpuw_vsvmvl() {
  // CHECK-LABEL: @test_vcmpuw_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpuw.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpuw_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvcmpu_vvvMvl() {
  // CHECK-LABEL: @test_pvcmpu_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvcmpu_vsvMvl() {
  // CHECK-LABEL: @test_pvcmpu_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmpu.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmpu_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vcmpswsx_vvvmvl() {
  // CHECK-LABEL: @test_vcmpswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswsx_vsvmvl() {
  // CHECK-LABEL: @test_vcmpswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vcmpswzx_vvvmvl() {
  // CHECK-LABEL: @test_vcmpswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vcmpswzx_vsvmvl() {
  // CHECK-LABEL: @test_vcmpswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvcmps_vvvMvl() {
  // CHECK-LABEL: @test_pvcmps_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvcmps_vsvMvl() {
  // CHECK-LABEL: @test_pvcmps_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcmps.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcmps_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vcmpsl_vvvmvl() {
  // CHECK-LABEL: @test_vcmpsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vcmpsl_vsvmvl() {
  // CHECK-LABEL: @test_vcmpsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcmpsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcmpsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmaxswsx_vvvmvl() {
  // CHECK-LABEL: @test_vmaxswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswsx_vsvmvl() {
  // CHECK-LABEL: @test_vmaxswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vmaxswzx_vvvmvl() {
  // CHECK-LABEL: @test_vmaxswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmaxswzx_vsvmvl() {
  // CHECK-LABEL: @test_vmaxswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvmaxs_vvvMvl() {
  // CHECK-LABEL: @test_pvmaxs_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvmaxs_vsvMvl() {
  // CHECK-LABEL: @test_pvmaxs_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmaxs.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmaxs_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vminswsx_vvvmvl() {
  // CHECK-LABEL: @test_vminswsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vminswsx_vsvmvl() {
  // CHECK-LABEL: @test_vminswsx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswsx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswsx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vminswzx_vvvmvl() {
  // CHECK-LABEL: @test_vminswzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vminswzx_vsvmvl() {
  // CHECK-LABEL: @test_vminswzx_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminswzx.vsvmvl(i32 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminswzx_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvmins_vvvMvl() {
  // CHECK-LABEL: @test_pvmins_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvmins_vsvMvl() {
  // CHECK-LABEL: @test_pvmins_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvmins.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvmins_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vmaxsl_vvvmvl() {
  // CHECK-LABEL: @test_vmaxsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmaxsl_vsvmvl() {
  // CHECK-LABEL: @test_vmaxsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmaxsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmaxsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vminsl_vvvmvl() {
  // CHECK-LABEL: @test_vminsl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vminsl_vsvmvl() {
  // CHECK-LABEL: @test_vminsl_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vminsl.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vminsl_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_vand_vvvmvl() {
  // CHECK-LABEL: @test_vand_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vand_vsvmvl() {
  // CHECK-LABEL: @test_vand_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vand.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vand_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvand_vvvMvl() {
  // CHECK-LABEL: @test_pvand_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvand_vsvMvl() {
  // CHECK-LABEL: @test_pvand_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvand.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvand_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vor_vvvmvl() {
  // CHECK-LABEL: @test_vor_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vor_vsvmvl() {
  // CHECK-LABEL: @test_vor_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vor.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vor_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvor_vvvMvl() {
  // CHECK-LABEL: @test_pvor_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvor_vsvMvl() {
  // CHECK-LABEL: @test_pvor_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvor.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvor_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vxor_vvvmvl() {
  // CHECK-LABEL: @test_vxor_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vxor_vsvmvl() {
  // CHECK-LABEL: @test_vxor_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vxor.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vxor_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pvxor_vvvMvl() {
  // CHECK-LABEL: @test_pvxor_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvxor_vsvMvl() {
  // CHECK-LABEL: @test_pvxor_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvxor.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvxor_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_veqv_vvvmvl() {
  // CHECK-LABEL: @test_veqv_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_veqv_vsvmvl() {
  // CHECK-LABEL: @test_veqv_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.veqv.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_veqv_vsvmvl(v1, vr2, vm1, vr3, 256);
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
test_pveqv_vvvMvl() {
  // CHECK-LABEL: @test_pveqv_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pveqv_vsvMvl() {
  // CHECK-LABEL: @test_pveqv_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pveqv.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pveqv_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vldz_vvmvl() {
  // CHECK-LABEL: @test_vldz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vldz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vldz_vvmvl(vr1, vm1, vr2, 256);
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
test_pvldzlo_vvmvl() {
  // CHECK-LABEL: @test_pvldzlo_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzlo.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzlo_vvmvl(vr1, vm1, vr2, 256);
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
test_pvldzup_vvmvl() {
  // CHECK-LABEL: @test_pvldzup_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldzup.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldzup_vvmvl(vr1, vm1, vr2, 256);
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
test_pvldz_vvMvl() {
  // CHECK-LABEL: @test_pvldz_vvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvldz.vvMvl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvldz_vvMvl(vr1, vm1_512, vr2, 256);
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
test_vpcnt_vvmvl() {
  // CHECK-LABEL: @test_vpcnt_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vpcnt.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vpcnt_vvmvl(vr1, vm1, vr2, 256);
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
test_pvpcntlo_vvmvl() {
  // CHECK-LABEL: @test_pvpcntlo_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntlo.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntlo_vvmvl(vr1, vm1, vr2, 256);
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
test_pvpcntup_vvmvl() {
  // CHECK-LABEL: @test_pvpcntup_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcntup.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcntup_vvmvl(vr1, vm1, vr2, 256);
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
test_pvpcnt_vvMvl() {
  // CHECK-LABEL: @test_pvpcnt_vvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvpcnt.vvMvl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvpcnt_vvMvl(vr1, vm1_512, vr2, 256);
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
test_vbrv_vvmvl() {
  // CHECK-LABEL: @test_vbrv_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vbrv.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vbrv_vvmvl(vr1, vm1, vr2, 256);
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
test_pvbrvlo_vvmvl() {
  // CHECK-LABEL: @test_pvbrvlo_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvlo.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvlo_vvmvl(vr1, vm1, vr2, 256);
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
test_pvbrvup_vvmvl() {
  // CHECK-LABEL: @test_pvbrvup_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrvup.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrvup_vvmvl(vr1, vm1, vr2, 256);
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
test_pvbrv_vvMvl() {
  // CHECK-LABEL: @test_pvbrv_vvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvbrv.vvMvl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvbrv_vvMvl(vr1, vm1_512, vr2, 256);
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
test_vsll_vvvmvl() {
  // CHECK-LABEL: @test_vsll_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsll_vvsmvl() {
  // CHECK-LABEL: @test_vsll_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsll.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsll_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_pvsll_vvvMvl() {
  // CHECK-LABEL: @test_pvsll_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsll_vvsMvl() {
  // CHECK-LABEL: @test_pvsll_vvsMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsll.vvsMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsll_vvsMvl(vr1, v2, vm1_512, vr3, 256);
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
test_vsrl_vvvmvl() {
  // CHECK-LABEL: @test_vsrl_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsrl_vvsmvl() {
  // CHECK-LABEL: @test_vsrl_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrl.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrl_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_pvsrl_vvvMvl() {
  // CHECK-LABEL: @test_pvsrl_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsrl_vvsMvl() {
  // CHECK-LABEL: @test_pvsrl_vvsMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsrl.vvsMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsrl_vvsMvl(vr1, v2, vm1_512, vr3, 256);
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
test_vslawsx_vvvmvl() {
  // CHECK-LABEL: @test_vslawsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vslawsx_vvsmvl() {
  // CHECK-LABEL: @test_vslawsx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawsx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawsx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vslawzx_vvvmvl() {
  // CHECK-LABEL: @test_vslawzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vslawzx_vvsmvl() {
  // CHECK-LABEL: @test_vslawzx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslawzx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslawzx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_pvsla_vvvMvl() {
  // CHECK-LABEL: @test_pvsla_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsla_vvsMvl() {
  // CHECK-LABEL: @test_pvsla_vvsMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsla.vvsMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsla_vvsMvl(vr1, v2, vm1_512, vr3, 256);
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
test_vslal_vvvmvl() {
  // CHECK-LABEL: @test_vslal_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vslal_vvsmvl() {
  // CHECK-LABEL: @test_vslal_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vslal.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vslal_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vsrawsx_vvvmvl() {
  // CHECK-LABEL: @test_vsrawsx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsrawsx_vvsmvl() {
  // CHECK-LABEL: @test_vsrawsx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawsx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawsx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vsrawzx_vvvmvl() {
  // CHECK-LABEL: @test_vsrawzx_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsrawzx_vvsmvl() {
  // CHECK-LABEL: @test_vsrawzx_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsrawzx.vvsmvl(<256 x double> %{{.*}}, i32 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsrawzx_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_pvsra_vvvMvl() {
  // CHECK-LABEL: @test_pvsra_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvsra_vvsMvl() {
  // CHECK-LABEL: @test_pvsra_vvsMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvsra.vvsMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvsra_vvsMvl(vr1, v2, vm1_512, vr3, 256);
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
test_vsral_vvvmvl() {
  // CHECK-LABEL: @test_vsral_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsral_vvsmvl() {
  // CHECK-LABEL: @test_vsral_vvsmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsral.vvsmvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsral_vvsmvl(vr1, v2, vm1, vr3, 256);
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
test_vsfa_vvssmvl() {
  // CHECK-LABEL: @test_vsfa_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsfa.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vsfa_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vfaddd_vvvmvl() {
  // CHECK-LABEL: @test_vfaddd_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfaddd_vsvmvl() {
  // CHECK-LABEL: @test_vfaddd_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfaddd.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfaddd_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfadds_vvvmvl() {
  // CHECK-LABEL: @test_vfadds_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfadds_vsvmvl() {
  // CHECK-LABEL: @test_vfadds_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfadds.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfadds_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfadd_vvvMvl() {
  // CHECK-LABEL: @test_pvfadd_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfadd_vsvMvl() {
  // CHECK-LABEL: @test_pvfadd_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfadd.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfadd_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfsubd_vvvmvl() {
  // CHECK-LABEL: @test_vfsubd_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfsubd_vsvmvl() {
  // CHECK-LABEL: @test_vfsubd_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubd.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubd_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfsubs_vvvmvl() {
  // CHECK-LABEL: @test_vfsubs_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfsubs_vsvmvl() {
  // CHECK-LABEL: @test_vfsubs_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsubs.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfsubs_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfsub_vvvMvl() {
  // CHECK-LABEL: @test_pvfsub_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfsub_vsvMvl() {
  // CHECK-LABEL: @test_pvfsub_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfsub.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfsub_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfmuld_vvvmvl() {
  // CHECK-LABEL: @test_vfmuld_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmuld_vsvmvl() {
  // CHECK-LABEL: @test_vfmuld_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuld.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuld_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfmuls_vvvmvl() {
  // CHECK-LABEL: @test_vfmuls_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmuls_vsvmvl() {
  // CHECK-LABEL: @test_vfmuls_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmuls.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmuls_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfmul_vvvMvl() {
  // CHECK-LABEL: @test_pvfmul_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfmul_vsvMvl() {
  // CHECK-LABEL: @test_pvfmul_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmul.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmul_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfdivd_vvvmvl() {
  // CHECK-LABEL: @test_vfdivd_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfdivd_vsvmvl() {
  // CHECK-LABEL: @test_vfdivd_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivd.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivd_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfdivs_vvvmvl() {
  // CHECK-LABEL: @test_vfdivs_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfdivs_vsvmvl() {
  // CHECK-LABEL: @test_vfdivs_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfdivs.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfdivs_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_vfcmpd_vvvmvl() {
  // CHECK-LABEL: @test_vfcmpd_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfcmpd_vsvmvl() {
  // CHECK-LABEL: @test_vfcmpd_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmpd.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmpd_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfcmps_vvvmvl() {
  // CHECK-LABEL: @test_vfcmps_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfcmps_vsvmvl() {
  // CHECK-LABEL: @test_vfcmps_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfcmps.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfcmps_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfcmp_vvvMvl() {
  // CHECK-LABEL: @test_pvfcmp_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfcmp_vsvMvl() {
  // CHECK-LABEL: @test_pvfcmp_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfcmp.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfcmp_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfmaxd_vvvmvl() {
  // CHECK-LABEL: @test_vfmaxd_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxd_vsvmvl() {
  // CHECK-LABEL: @test_vfmaxd_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxd.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxd_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfmaxs_vvvmvl() {
  // CHECK-LABEL: @test_vfmaxs_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmaxs_vsvmvl() {
  // CHECK-LABEL: @test_vfmaxs_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmaxs.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmaxs_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfmax_vvvMvl() {
  // CHECK-LABEL: @test_pvfmax_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfmax_vsvMvl() {
  // CHECK-LABEL: @test_pvfmax_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmax.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmax_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfmind_vvvmvl() {
  // CHECK-LABEL: @test_vfmind_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmind_vsvmvl() {
  // CHECK-LABEL: @test_vfmind_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmind.vsvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmind_vsvmvl(vd1, vr2, vm1, vr3, 256);
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
test_vfmins_vvvmvl() {
  // CHECK-LABEL: @test_vfmins_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vfmins_vsvmvl() {
  // CHECK-LABEL: @test_vfmins_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmins.vsvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vfmins_vsvmvl(vf1, vr2, vm1, vr3, 256);
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
test_pvfmin_vvvMvl() {
  // CHECK-LABEL: @test_pvfmin_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_pvfmin_vsvMvl() {
  // CHECK-LABEL: @test_pvfmin_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmin.vsvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvfmin_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vfmadd_vvvvmvl() {
  // CHECK-LABEL: @test_vfmadd_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmadd_vsvvmvl() {
  // CHECK-LABEL: @test_vfmadd_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vsvvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vsvvmvl(vd1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmadd_vvsvmvl() {
  // CHECK-LABEL: @test_vfmadd_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmadd.vvsvmvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmadd_vvsvmvl(vr1, vd1, vr3, vm1, vr4, 256);
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
test_vfmads_vvvvmvl() {
  // CHECK-LABEL: @test_vfmads_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmads_vsvvmvl() {
  // CHECK-LABEL: @test_vfmads_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vsvvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vsvvmvl(vf1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmads_vvsvmvl() {
  // CHECK-LABEL: @test_vfmads_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmads.vvsvmvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmads_vvsvmvl(vr1, vf1, vr3, vm1, vr4, 256);
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
test_pvfmad_vvvvMvl() {
  // CHECK-LABEL: @test_pvfmad_vvvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvvvMvl(vr1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfmad_vsvvMvl() {
  // CHECK-LABEL: @test_pvfmad_vsvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vsvvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vsvvMvl(v1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfmad_vvsvMvl() {
  // CHECK-LABEL: @test_pvfmad_vvsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmad.vvsvMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmad_vvsvMvl(vr1, v1, vr3, vm1_512, vr4, 256);
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
test_vfmsbd_vvvvmvl() {
  // CHECK-LABEL: @test_vfmsbd_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbd_vsvvmvl() {
  // CHECK-LABEL: @test_vfmsbd_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vsvvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vsvvmvl(vd1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbd_vvsvmvl() {
  // CHECK-LABEL: @test_vfmsbd_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbd.vvsvmvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbd_vvsvmvl(vr1, vd1, vr3, vm1, vr4, 256);
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
test_vfmsbs_vvvvmvl() {
  // CHECK-LABEL: @test_vfmsbs_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbs_vsvvmvl() {
  // CHECK-LABEL: @test_vfmsbs_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vsvvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vsvvmvl(vf1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfmsbs_vvsvmvl() {
  // CHECK-LABEL: @test_vfmsbs_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfmsbs.vvsvmvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfmsbs_vvsvmvl(vr1, vf1, vr3, vm1, vr4, 256);
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
test_pvfmsb_vvvvMvl() {
  // CHECK-LABEL: @test_pvfmsb_vvvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvvvMvl(vr1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfmsb_vsvvMvl() {
  // CHECK-LABEL: @test_pvfmsb_vsvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vsvvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vsvvMvl(v1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfmsb_vvsvMvl() {
  // CHECK-LABEL: @test_pvfmsb_vvsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfmsb.vvsvMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfmsb_vvsvMvl(vr1, v1, vr3, vm1_512, vr4, 256);
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
test_vfnmadd_vvvvmvl() {
  // CHECK-LABEL: @test_vfnmadd_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmadd_vsvvmvl() {
  // CHECK-LABEL: @test_vfnmadd_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vsvvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vsvvmvl(vd1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmadd_vvsvmvl() {
  // CHECK-LABEL: @test_vfnmadd_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmadd.vvsvmvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmadd_vvsvmvl(vr1, vd1, vr3, vm1, vr4, 256);
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
test_vfnmads_vvvvmvl() {
  // CHECK-LABEL: @test_vfnmads_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmads_vsvvmvl() {
  // CHECK-LABEL: @test_vfnmads_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vsvvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vsvvmvl(vf1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmads_vvsvmvl() {
  // CHECK-LABEL: @test_vfnmads_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmads.vvsvmvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmads_vvsvmvl(vr1, vf1, vr3, vm1, vr4, 256);
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
test_pvfnmad_vvvvMvl() {
  // CHECK-LABEL: @test_pvfnmad_vvvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvvvMvl(vr1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmad_vsvvMvl() {
  // CHECK-LABEL: @test_pvfnmad_vsvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vsvvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vsvvMvl(v1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmad_vvsvMvl() {
  // CHECK-LABEL: @test_pvfnmad_vvsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmad.vvsvMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmad_vvsvMvl(vr1, v1, vr3, vm1_512, vr4, 256);
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
test_vfnmsbd_vvvvmvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vsvvmvl() {
  // CHECK-LABEL: @test_vfnmsbd_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vsvvmvl(double %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vsvvmvl(vd1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbd_vvsvmvl() {
  // CHECK-LABEL: @test_vfnmsbd_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbd.vvsvmvl(<256 x double> %{{.*}}, double %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbd_vvsvmvl(vr1, vd1, vr3, vm1, vr4, 256);
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
test_vfnmsbs_vvvvmvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvvvmvl(vr1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vsvvmvl() {
  // CHECK-LABEL: @test_vfnmsbs_vsvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vsvvmvl(float %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vsvvmvl(vf1, vr2, vr3, vm1, vr4, 256);
}

void __attribute__((noinline))
test_vfnmsbs_vvsvmvl() {
  // CHECK-LABEL: @test_vfnmsbs_vvsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfnmsbs.vvsvmvl(<256 x double> %{{.*}}, float %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_vfnmsbs_vvsvmvl(vr1, vf1, vr3, vm1, vr4, 256);
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
test_pvfnmsb_vvvvMvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvvvMvl(vr1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vsvvMvl() {
  // CHECK-LABEL: @test_pvfnmsb_vsvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vsvvMvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vsvvMvl(v1, vr2, vr3, vm1_512, vr4, 256);
}

void __attribute__((noinline))
test_pvfnmsb_vvsvMvl() {
  // CHECK-LABEL: @test_pvfnmsb_vvsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvfnmsb.vvsvMvl(<256 x double> %{{.*}}, i64 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr4 = _vel_pvfnmsb_vvsvMvl(vr1, v1, vr3, vm1_512, vr4, 256);
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
test_vcvtwdsx_vvmvl() {
  // CHECK-LABEL: @test_vcvtwdsx_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsx.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdsx_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwdsxrz_vvmvl() {
  // CHECK-LABEL: @test_vcvtwdsxrz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdsxrz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdsxrz_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwdzx_vvmvl() {
  // CHECK-LABEL: @test_vcvtwdzx_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzx.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdzx_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwdzxrz_vvmvl() {
  // CHECK-LABEL: @test_vcvtwdzxrz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwdzxrz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwdzxrz_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwssx_vvmvl() {
  // CHECK-LABEL: @test_vcvtwssx_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssx.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwssx_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwssxrz_vvmvl() {
  // CHECK-LABEL: @test_vcvtwssxrz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwssxrz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwssxrz_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwszx_vvmvl() {
  // CHECK-LABEL: @test_vcvtwszx_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszx.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwszx_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtwszxrz_vvmvl() {
  // CHECK-LABEL: @test_vcvtwszxrz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtwszxrz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtwszxrz_vvmvl(vr1, vm1, vr2, 256);
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
test_pvcvtws_vvMvl() {
  // CHECK-LABEL: @test_pvcvtws_vvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtws.vvMvl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcvtws_vvMvl(vr1, vm1_512, vr2, 256);
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
test_pvcvtwsrz_vvMvl() {
  // CHECK-LABEL: @test_pvcvtwsrz_vvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.pvcvtwsrz.vvMvl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_pvcvtwsrz_vvMvl(vr1, vm1_512, vr2, 256);
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
test_vcvtld_vvmvl() {
  // CHECK-LABEL: @test_vcvtld_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtld.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtld_vvmvl(vr1, vm1, vr2, 256);
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
test_vcvtldrz_vvmvl() {
  // CHECK-LABEL: @test_vcvtldrz_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcvtldrz.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcvtldrz_vvmvl(vr1, vm1, vr2, 256);
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
test_vmrg_vvvml() {
  // CHECK-LABEL: @test_vmrg_vvvml
  // CHECK: call <256 x double> @llvm.ve.vl.vmrg.vvvml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vmrg_vvvml(vr1, vr2, vm1, 256);
}

void __attribute__((noinline))
test_vmrg_vvvmvl() {
  // CHECK-LABEL: @test_vmrg_vvvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrg.vvvmvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmrg_vvvmvl(vr1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmrg_vsvml() {
  // CHECK-LABEL: @test_vmrg_vsvml
  // CHECK: call <256 x double> @llvm.ve.vl.vmrg.vsvml(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vmrg_vsvml(v1, vr2, vm1, 256);
}

void __attribute__((noinline))
test_vmrg_vsvmvl() {
  // CHECK-LABEL: @test_vmrg_vsvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrg.vsvmvl(i64 %{{.*}}, <256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmrg_vsvmvl(v1, vr2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vmrgw_vvvMl() {
  // CHECK-LABEL: @test_vmrgw_vvvMl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrgw.vvvMl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vmrgw_vvvMl(vr1, vr2, vm1_512, 256);
}

void __attribute__((noinline))
test_vmrgw_vvvMvl() {
  // CHECK-LABEL: @test_vmrgw_vvvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrgw.vvvMvl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmrgw_vvvMvl(vr1, vr2, vm1_512, vr3, 256);
}

void __attribute__((noinline))
test_vmrgw_vsvMl() {
  // CHECK-LABEL: @test_vmrgw_vsvMl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrgw.vsvMl(i32 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vmrgw_vsvMl(v1, vr2, vm1_512, 256);
}

void __attribute__((noinline))
test_vmrgw_vsvMvl() {
  // CHECK-LABEL: @test_vmrgw_vsvMvl
  // CHECK: call <256 x double> @llvm.ve.vl.vmrgw.vsvMvl(i32 %{{.*}}, <256 x double> %{{.*}}, <512 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vmrgw_vsvMvl(v1, vr2, vm1_512, vr3, 256);
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
test_vcp_vvmvl() {
  // CHECK-LABEL: @test_vcp_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vcp.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vcp_vvmvl(vr1, vm1, vr2, 256);
}

void __attribute__((noinline))
test_vex_vvmvl() {
  // CHECK-LABEL: @test_vex_vvmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vex.vvmvl(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vex_vvmvl(vr1, vm1, vr2, 256);
}

void __attribute__((noinline))
test_vfmklat_ml(int vl) {
  // CHECK-LABEL: @test_vfmklat_ml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklat.ml(i32 %{{.*}})
  vm1 = _vel_vfmklat_ml(vl);
}

void __attribute__((noinline))
test_vfmklaf_ml(int vl) {
  // CHECK-LABEL: @test_vfmklaf_ml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklaf.ml(i32 %{{.*}})
  vm1 = _vel_vfmklaf_ml(vl);
}

void __attribute__((noinline))
test_pvfmkat_Ml(int vl) {
  // CHECK-LABEL: @test_pvfmkat_Ml
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkat.Ml(i32 %{{.*}})
  vm1_512 = _vel_pvfmkat_Ml(vl);
}

void __attribute__((noinline))
test_pvfmkaf_Ml(int vl) {
  // CHECK-LABEL: @test_pvfmkaf_Ml
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkaf.Ml(i32 %{{.*}})
  vm1_512 = _vel_pvfmkaf_Ml(vl);
}

void __attribute__((noinline))
test_vfmklgt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklgt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkllt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkllt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkllt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkllt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkllt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkllt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkllt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkllt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklne_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklne_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkleq_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkleq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkleq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkleq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkleq_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkleq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkleq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkleq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklge_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklge_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklle_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklle_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklle.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklle_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklle_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklle_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklle.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklle_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklnum_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklnum_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklltnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklltnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklnenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklnenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkleqnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkleqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkleqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkleqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkleqnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkleqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkleqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkleqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmklgenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmklgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmklgenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmklgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmklgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmklgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkllenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkllenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkllenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkllenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkllenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkllenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkllenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkllenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwgt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwgt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwlt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwlt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwlt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwlt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwlt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwlt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwlt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwlt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwne_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwne_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkweq_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkweq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkweq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkweq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkweq_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkweq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkweq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkweq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwge_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwge_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwle_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwle_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwle.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwle_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwle_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwle_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwle.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwle_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwnum_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwnum_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwltnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwltnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwnenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwnenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkweqnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkweqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkweqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkweqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkweqnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkweqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkweqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkweqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwgenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwgenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkwlenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkwlenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwlenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwlenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkwlenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkwlenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkwlenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkwlenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlogt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupgt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlogt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupgt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlolt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlolt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlolt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwuplt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwuplt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuplt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlolt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlolt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlolt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlolt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwuplt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwuplt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuplt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuplt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlone_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlone_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlone.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlone_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupne_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlone_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlone_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlone.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlone_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupne_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwloeq_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwloeq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloeq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupeq_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupeq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupeq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwloeq_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwloeq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloeq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloeq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupeq_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupeq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupeq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupeq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwloge_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwloge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupge_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwloge_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwloge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupge_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlole_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlole_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlole.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlole_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwuple_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwuple_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuple.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuple_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlole_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlole_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlole.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlole_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwuple_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwuple_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuple.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuple_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlonum_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupnum_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlonum_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupnum_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlonan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlonan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlogtnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlogtnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwloltnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwloltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupltnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwloltnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwloltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupltnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlonenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupnenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlonenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlonenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlonenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlonenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupnenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwloeqnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwloeqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloeqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupeqnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupeqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupeqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwloeqnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwloeqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwloeqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwloeqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupeqnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupeqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupeqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupeqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlogenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwupgenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlogenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlogenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlogenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlogenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwupgenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwupgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwupgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwupgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwlolenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlolenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlolenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwuplenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwuplenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuplenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlolenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwlolenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwlolenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwlolenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwuplenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkwuplenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkwuplenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkwuplenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkwgt_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgt_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgt.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgt_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwgt_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgt_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgt.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgt_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwlt_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlt_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwlt.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwlt_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlt_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlt_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwlt.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwlt_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwne_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwne_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwne.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwne_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwne_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwne_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwne.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwne_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkweq_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkweq_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkweq.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkweq_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkweq_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkweq_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkweq.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkweq_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwge_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwge_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwge.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwge_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwge_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwge_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwge.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwge_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwle_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwle_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwle.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwle_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwle_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwle_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwle.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwle_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwnum_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnum_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnum.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnum_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwnum_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnum_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnum.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnum_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwgtnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgtnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgtnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgtnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwgtnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgtnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgtnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgtnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwltnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwltnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwltnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwltnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwltnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwltnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwltnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwltnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwnenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwnenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwnenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwnenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwnenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkweqnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkweqnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkweqnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkweqnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkweqnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkweqnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkweqnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkweqnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwgenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwgenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwgenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwgenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwgenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkwlenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwlenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwlenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkwlenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkwlenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkwlenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkwlenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_vfmkdgt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdgt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdlt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdlt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdlt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdlt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdlt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdlt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdlt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdlt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdne_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdne_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdeq_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdeq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdeq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdeq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdeq_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdeq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdeq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdeq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdge_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdge_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdle_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdle_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdle.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdle_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdle_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdle_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdle.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdle_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdnum_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdnum_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdltnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdltnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdnenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdnenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdeqnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdeqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdeqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdeqnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdeqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdeqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdeqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdgenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdgenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkdlenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkdlenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdlenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdlenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkdlenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkdlenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkdlenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkdlenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksgt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksgt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkslt_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkslt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkslt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkslt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkslt_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkslt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkslt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkslt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksne_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksne_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkseq_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkseq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkseq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkseq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkseq_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkseq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkseq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkseq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksge_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksge_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksle_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksle_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksle.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksle_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksle_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksle_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksle.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksle_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksnum_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksnum_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksltnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksltnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksnenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksnenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkseqnan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkseqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkseqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkseqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkseqnan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkseqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkseqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkseqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmksgenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmksgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmksgenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmksgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmksgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmksgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_vfmkslenan_mvl(int vl) {
  // CHECK-LABEL: @test_vfmkslenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkslenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkslenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_vfmkslenan_mvml(int vl) {
  // CHECK-LABEL: @test_vfmkslenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.vfmkslenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_vfmkslenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslogt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslogt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupgt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupgt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslogt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslogt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupgt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupgt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslolt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslolt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslolt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslolt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksuplt_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksuplt_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuplt.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuplt_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslolt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslolt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslolt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslolt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksuplt_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksuplt_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuplt.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuplt_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslone_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslone_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslone.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslone_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupne_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupne_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupne.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupne_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslone_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslone_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslone.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslone_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupne_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupne_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupne.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupne_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksloeq_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksloeq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloeq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloeq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupeq_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupeq_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupeq.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupeq_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksloeq_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksloeq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloeq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloeq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupeq_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupeq_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupeq.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupeq_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksloge_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksloge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupge_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupge_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupge.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupge_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksloge_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksloge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupge_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupge_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupge.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupge_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslole_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslole_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslole.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslole_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksuple_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksuple_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuple.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuple_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslole_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslole_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslole.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslole_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksuple_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksuple_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuple.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuple_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslonum_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslonum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupnum_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupnum_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnum.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnum_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslonum_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslonum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupnum_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupnum_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnum.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnum_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslonan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslonan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslonan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslonan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslogtnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslogtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupgtnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupgtnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgtnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslogtnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslogtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupgtnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupgtnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgtnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgtnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksloltnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksloltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupltnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupltnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupltnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksloltnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksloltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupltnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupltnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupltnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupltnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslonenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslonenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupnenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupnenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslonenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslonenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslonenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslonenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupnenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupnenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupnenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupnenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksloeqnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksloeqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloeqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupeqnan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupeqnan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupeqnan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksloeqnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksloeqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksloeqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksloeqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupeqnan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupeqnan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupeqnan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupeqnan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslogenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslogenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksupgenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksupgenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslogenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslogenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslogenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslogenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksupgenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksupgenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksupgenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksupgenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmkslolenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslolenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslolenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksuplenan_mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksuplenan_mvl
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuplenan_mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslolenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmkslolenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmkslolenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmkslolenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksuplenan_mvml(int vl) {
  // CHECK-LABEL: @test_pvfmksuplenan_mvml
  // CHECK: call <256 x i1> @llvm.ve.vl.pvfmksuplenan.mvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 %{{.*}})
  vm1 = _vel_pvfmksuplenan_mvml(vr1, vm2, vl);
}

void __attribute__((noinline))
test_pvfmksgt_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksgt_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgt.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgt_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksgt_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksgt_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgt.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgt_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkslt_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslt_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkslt.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkslt_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslt_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkslt_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkslt.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkslt_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksne_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksne_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksne.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksne_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksne_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksne_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksne.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksne_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkseq_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkseq_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkseq.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkseq_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkseq_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkseq_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkseq.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkseq_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksge_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksge_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksge.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksge_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksge_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksge_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksge.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksge_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksle_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksle_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksle.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksle_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksle_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksle_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksle.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksle_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksnum_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksnum_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnum.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnum_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksnum_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksnum_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnum.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnum_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksgtnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksgtnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgtnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgtnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksgtnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksgtnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgtnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgtnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksltnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksltnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksltnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksltnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksltnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksltnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksltnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksltnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksnenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksnenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksnenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksnenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksnenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksnenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkseqnan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkseqnan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkseqnan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkseqnan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkseqnan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkseqnan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkseqnan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkseqnan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmksgenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmksgenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmksgenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmksgenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmksgenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmksgenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_pvfmkslenan_Mvl(int vl) {
  // CHECK-LABEL: @test_pvfmkslenan_Mvl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkslenan.Mvl(<256 x double> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkslenan_Mvl(vr1, vl);
}

void __attribute__((noinline))
test_pvfmkslenan_MvMl(int vl) {
  // CHECK-LABEL: @test_pvfmkslenan_MvMl
  // CHECK: call <512 x i1> @llvm.ve.vl.pvfmkslenan.MvMl(<256 x double> %{{.*}}, <512 x i1> %{{.*}}, i32 %{{.*}})
  vm1_512 = _vel_pvfmkslenan_MvMl(vr1, vm2_512, vl);
}

void __attribute__((noinline))
test_vsumwsx_vvl() {
  // CHECK-LABEL: @test_vsumwsx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwsx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsumwsx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsumwsx_vvml() {
  // CHECK-LABEL: @test_vsumwsx_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwsx.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr2 = _vel_vsumwsx_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vsumwzx_vvl() {
  // CHECK-LABEL: @test_vsumwzx_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwzx.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsumwzx_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsumwzx_vvml() {
  // CHECK-LABEL: @test_vsumwzx_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vsumwzx.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr2 = _vel_vsumwzx_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vsuml_vvl() {
  // CHECK-LABEL: @test_vsuml_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vsuml.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vsuml_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vsuml_vvml() {
  // CHECK-LABEL: @test_vsuml_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vsuml.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr2 = _vel_vsuml_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vfsumd_vvl() {
  // CHECK-LABEL: @test_vfsumd_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsumd.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsumd_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsumd_vvml() {
  // CHECK-LABEL: @test_vfsumd_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vfsumd.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr2 = _vel_vfsumd_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vfsums_vvl() {
  // CHECK-LABEL: @test_vfsums_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vfsums.vvl(<256 x double> %{{.*}}, i32 256)
  vr2 = _vel_vfsums_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vfsums_vvml() {
  // CHECK-LABEL: @test_vfsums_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vfsums.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr2 = _vel_vfsums_vvml(vr1, vm1, 256);
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
test_vrand_vvml() {
  // CHECK-LABEL: @test_vrand_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vrand.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vrand_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vror_vvl() {
  // CHECK-LABEL: @test_vror_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vror.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vror_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vror_vvml() {
  // CHECK-LABEL: @test_vror_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vror.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vror_vvml(vr1, vm1, 256);
}

void __attribute__((noinline))
test_vrxor_vvl() {
  // CHECK-LABEL: @test_vrxor_vvl
  // CHECK: call <256 x double> @llvm.ve.vl.vrxor.vvl(<256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vrxor_vvl(vr1, 256);
}

void __attribute__((noinline))
test_vrxor_vvml() {
  // CHECK-LABEL: @test_vrxor_vvml
  // CHECK: call <256 x double> @llvm.ve.vl.vrxor.vvml(<256 x double> %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vrxor_vvml(vr1, vm1, 256);
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
test_vgt_vvssml() {
  // CHECK-LABEL: @test_vgt_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgt.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgt_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgt_vvssmvl() {
  // CHECK-LABEL: @test_vgt_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgt.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgt_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtnc_vvssml() {
  // CHECK-LABEL: @test_vgtnc_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtnc.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtnc_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtnc_vvssmvl() {
  // CHECK-LABEL: @test_vgtnc_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtnc.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtnc_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtu_vvssml() {
  // CHECK-LABEL: @test_vgtu_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtu.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtu_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtu_vvssmvl() {
  // CHECK-LABEL: @test_vgtu_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtu.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtu_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtunc_vvssml() {
  // CHECK-LABEL: @test_vgtunc_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtunc.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtunc_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtunc_vvssmvl() {
  // CHECK-LABEL: @test_vgtunc_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtunc.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtunc_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtlsx_vvssml() {
  // CHECK-LABEL: @test_vgtlsx_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsx.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsx_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtlsx_vvssmvl() {
  // CHECK-LABEL: @test_vgtlsx_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsx.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsx_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtlsxnc_vvssml() {
  // CHECK-LABEL: @test_vgtlsxnc_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsxnc.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsxnc_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtlsxnc_vvssmvl() {
  // CHECK-LABEL: @test_vgtlsxnc_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlsxnc.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlsxnc_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtlzx_vvssml() {
  // CHECK-LABEL: @test_vgtlzx_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzx.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzx_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtlzx_vvssmvl() {
  // CHECK-LABEL: @test_vgtlzx_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzx.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzx_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
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
test_vgtlzxnc_vvssml() {
  // CHECK-LABEL: @test_vgtlzxnc_vvssml
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzxnc.vvssml(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzxnc_vvssml(vr1, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vgtlzxnc_vvssmvl() {
  // CHECK-LABEL: @test_vgtlzxnc_vvssmvl
  // CHECK: call <256 x double> @llvm.ve.vl.vgtlzxnc.vvssmvl(<256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, <256 x double> %{{.*}}, i32 256)
  vr3 = _vel_vgtlzxnc_vvssmvl(vr1, v1, v2, vm1, vr3, 256);
}

void __attribute__((noinline))
test_vsc_vvssl() {
  // CHECK-LABEL: @test_vsc_vvssl
  // CHECK: call void @llvm.ve.vl.vsc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsc_vvssml() {
  // CHECK-LABEL: @test_vsc_vvssml
  // CHECK: call void @llvm.ve.vl.vsc.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vsc_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscnc_vvssl() {
  // CHECK-LABEL: @test_vscnc_vvssl
  // CHECK: call void @llvm.ve.vl.vscnc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscnc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscnc_vvssml() {
  // CHECK-LABEL: @test_vscnc_vvssml
  // CHECK: call void @llvm.ve.vl.vscnc.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscnc_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscot_vvssl() {
  // CHECK-LABEL: @test_vscot_vvssl
  // CHECK: call void @llvm.ve.vl.vscot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscot_vvssml() {
  // CHECK-LABEL: @test_vscot_vvssml
  // CHECK: call void @llvm.ve.vl.vscot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscncot_vvssl() {
  // CHECK-LABEL: @test_vscncot_vvssl
  // CHECK: call void @llvm.ve.vl.vscncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscncot_vvssml() {
  // CHECK-LABEL: @test_vscncot_vvssml
  // CHECK: call void @llvm.ve.vl.vscncot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscncot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscu_vvssl() {
  // CHECK-LABEL: @test_vscu_vvssl
  // CHECK: call void @llvm.ve.vl.vscu.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscu_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscu_vvssml() {
  // CHECK-LABEL: @test_vscu_vvssml
  // CHECK: call void @llvm.ve.vl.vscu.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscu_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscunc_vvssl() {
  // CHECK-LABEL: @test_vscunc_vvssl
  // CHECK: call void @llvm.ve.vl.vscunc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscunc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscunc_vvssml() {
  // CHECK-LABEL: @test_vscunc_vvssml
  // CHECK: call void @llvm.ve.vl.vscunc.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscunc_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscuot_vvssl() {
  // CHECK-LABEL: @test_vscuot_vvssl
  // CHECK: call void @llvm.ve.vl.vscuot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscuot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscuot_vvssml() {
  // CHECK-LABEL: @test_vscuot_vvssml
  // CHECK: call void @llvm.ve.vl.vscuot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscuot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscuncot_vvssl() {
  // CHECK-LABEL: @test_vscuncot_vvssl
  // CHECK: call void @llvm.ve.vl.vscuncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscuncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscuncot_vvssml() {
  // CHECK-LABEL: @test_vscuncot_vvssml
  // CHECK: call void @llvm.ve.vl.vscuncot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscuncot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vscl_vvssl() {
  // CHECK-LABEL: @test_vscl_vvssl
  // CHECK: call void @llvm.ve.vl.vscl.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vscl_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vscl_vvssml() {
  // CHECK-LABEL: @test_vscl_vvssml
  // CHECK: call void @llvm.ve.vl.vscl.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vscl_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vsclnc_vvssl() {
  // CHECK-LABEL: @test_vsclnc_vvssl
  // CHECK: call void @llvm.ve.vl.vsclnc.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclnc_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclnc_vvssml() {
  // CHECK-LABEL: @test_vsclnc_vvssml
  // CHECK: call void @llvm.ve.vl.vsclnc.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vsclnc_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vsclot_vvssl() {
  // CHECK-LABEL: @test_vsclot_vvssl
  // CHECK: call void @llvm.ve.vl.vsclot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclot_vvssml() {
  // CHECK-LABEL: @test_vsclot_vvssml
  // CHECK: call void @llvm.ve.vl.vsclot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vsclot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_vsclncot_vvssl() {
  // CHECK-LABEL: @test_vsclncot_vvssl
  // CHECK: call void @llvm.ve.vl.vsclncot.vvssl(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 256)
  _vel_vsclncot_vvssl(vr1, vr2, v1, v2, 256);
}

void __attribute__((noinline))
test_vsclncot_vvssml() {
  // CHECK-LABEL: @test_vsclncot_vvssml
  // CHECK: call void @llvm.ve.vl.vsclncot.vvssml(<256 x double> %{{.*}}, <256 x double> %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, <256 x i1> %{{.*}}, i32 256)
  _vel_vsclncot_vvssml(vr1, vr2, v1, v2, vm1, 256);
}

void __attribute__((noinline))
test_andm_mmm() {
  // CHECK-LABEL: @test_andm_mmm
  // CHECK: call <256 x i1> @llvm.ve.vl.andm.mmm(<256 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm3 = _vel_andm_mmm(vm1, vm2);
}

void __attribute__((noinline))
test_andm_MMM() {
  // CHECK-LABEL: @test_andm_MMM
  // CHECK: call <512 x i1> @llvm.ve.vl.andm.MMM(<512 x i1> %{{.*}}, <512 x i1> %{{.*}})
  vm3_512 = _vel_andm_MMM(vm1_512, vm2_512);
}

void __attribute__((noinline))
test_orm_mmm() {
  // CHECK-LABEL: @test_orm_mmm
  // CHECK: call <256 x i1> @llvm.ve.vl.orm.mmm(<256 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm3 = _vel_orm_mmm(vm1, vm2);
}

void __attribute__((noinline))
test_orm_MMM() {
  // CHECK-LABEL: @test_orm_MMM
  // CHECK: call <512 x i1> @llvm.ve.vl.orm.MMM(<512 x i1> %{{.*}}, <512 x i1> %{{.*}})
  vm3_512 = _vel_orm_MMM(vm1_512, vm2_512);
}

void __attribute__((noinline))
test_xorm_mmm() {
  // CHECK-LABEL: @test_xorm_mmm
  // CHECK: call <256 x i1> @llvm.ve.vl.xorm.mmm(<256 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm3 = _vel_xorm_mmm(vm1, vm2);
}

void __attribute__((noinline))
test_xorm_MMM() {
  // CHECK-LABEL: @test_xorm_MMM
  // CHECK: call <512 x i1> @llvm.ve.vl.xorm.MMM(<512 x i1> %{{.*}}, <512 x i1> %{{.*}})
  vm3_512 = _vel_xorm_MMM(vm1_512, vm2_512);
}

void __attribute__((noinline))
test_eqvm_mmm() {
  // CHECK-LABEL: @test_eqvm_mmm
  // CHECK: call <256 x i1> @llvm.ve.vl.eqvm.mmm(<256 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm3 = _vel_eqvm_mmm(vm1, vm2);
}

void __attribute__((noinline))
test_eqvm_MMM() {
  // CHECK-LABEL: @test_eqvm_MMM
  // CHECK: call <512 x i1> @llvm.ve.vl.eqvm.MMM(<512 x i1> %{{.*}}, <512 x i1> %{{.*}})
  vm3_512 = _vel_eqvm_MMM(vm1_512, vm2_512);
}

void __attribute__((noinline))
test_nndm_mmm() {
  // CHECK-LABEL: @test_nndm_mmm
  // CHECK: call <256 x i1> @llvm.ve.vl.nndm.mmm(<256 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm3 = _vel_nndm_mmm(vm1, vm2);
}

void __attribute__((noinline))
test_nndm_MMM() {
  // CHECK-LABEL: @test_nndm_MMM
  // CHECK: call <512 x i1> @llvm.ve.vl.nndm.MMM(<512 x i1> %{{.*}}, <512 x i1> %{{.*}})
  vm3_512 = _vel_nndm_MMM(vm1_512, vm2_512);
}

void __attribute__((noinline))
test_negm_mm() {
  // CHECK-LABEL: @test_negm_mm
  // CHECK: call <256 x i1> @llvm.ve.vl.negm.mm(<256 x i1> %{{.*}})
  vm2 = _vel_negm_mm(vm1);
}

void __attribute__((noinline))
test_negm_MM() {
  // CHECK-LABEL: @test_negm_MM
  // CHECK: call <512 x i1> @llvm.ve.vl.negm.MM(<512 x i1> %{{.*}})
  vm2_512 = _vel_negm_MM(vm1_512);
}

void __attribute__((noinline))
test_pcvm_sml() {
  // CHECK-LABEL: @test_pcvm_sml
  // CHECK: call i64 @llvm.ve.vl.pcvm.sml(<256 x i1> %{{.*}}, i32 256)
  v1 = _vel_pcvm_sml(vm1, 256);
}

void __attribute__((noinline))
test_lzvm_sml() {
  // CHECK-LABEL: @test_lzvm_sml
  // CHECK: call i64 @llvm.ve.vl.lzvm.sml(<256 x i1> %{{.*}}, i32 256)
  v1 = _vel_lzvm_sml(vm1, 256);
}

void __attribute__((noinline))
test_tovm_sml() {
  // CHECK-LABEL: @test_tovm_sml
  // CHECK: call i64 @llvm.ve.vl.tovm.sml(<256 x i1> %{{.*}}, i32 256)
  v1 = _vel_tovm_sml(vm1, 256);
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

void __attribute__((noinline))
test_pack_f32p(float* p1, float* p2) {
  // CHECK-LABEL: @test_pack_f32p
  // CHECK: call i64 @llvm.ve.vl.pack.f32p(ptr %{{.*}}, ptr %{{.*}})
  v1 = _vel_pack_f32p(p1, p2);
}

void __attribute__((noinline))
test_pack_f32a(float* p) {
  // CHECK-LABEL: @test_pack_f32a
  // CHECK: call i64 @llvm.ve.vl.pack.f32a(ptr %{{.*}})
  v1 = _vel_pack_f32a(p);
}

void __attribute__((noinline))
test_extract_vm512u() {
  // CHECK-LABEL: @test_extract_vm512u
  // CHECK: call <256 x i1> @llvm.ve.vl.extract.vm512u(<512 x i1> %{{.*}})
  vm1 = _vel_extract_vm512u(vm1_512);
}

void __attribute__((noinline))
test_extract_vm512l() {
  // CHECK-LABEL: @test_extract_vm512l
  // CHECK: call <256 x i1> @llvm.ve.vl.extract.vm512l(<512 x i1> %{{.*}})
  vm1 = _vel_extract_vm512l(vm1_512);
}

void __attribute__((noinline))
test_insert_vm512u() {
  // CHECK-LABEL: @test_insert_vm512u
  // CHECK: call <512 x i1> @llvm.ve.vl.insert.vm512u(<512 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm1_512 = _vel_insert_vm512u(vm1_512, vm1);
}

void __attribute__((noinline))
test_insert_vm512l() {
  // CHECK-LABEL: @test_insert_vm512l
  // CHECK: call <512 x i1> @llvm.ve.vl.insert.vm512l(<512 x i1> %{{.*}}, <256 x i1> %{{.*}})
  vm1_512 = _vel_insert_vm512l(vm1_512, vm1);
}
