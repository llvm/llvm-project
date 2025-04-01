// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_floor_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.floor.f32(
float4 test_lit_double(double NDotL, double NDotH, double M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_floor_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.floor.f32(
float4 test_lit_int(int NDotL, int NDotH, int M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_floor_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.floor.f32(
float4 test_lit_uint(uint NDotL, uint NDotH, uint M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_floor_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.floor.f32(
float4 test_lit_int64_t(int64_t NDotL, int64_t NDotH, int64_t M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_floor_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.floor.f32(
float4 test_lit_uint64_t(uint64_t NDotL, uint64_t NDotH, uint64_t M) { return lit(NDotL, NDotH, M); }
