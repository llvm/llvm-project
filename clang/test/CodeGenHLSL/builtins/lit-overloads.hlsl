// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litEddd(
// CHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litIfEEKNS_8__detail9enable_ifIXaasr8__detail13is_arithmeticIT_EE5Valuesr8__detail7is_sameIfS3_EE5valueEDv4_S3_E4TypeES3_S3_S3_(
float4 test_lit_double(double NDotL, double NDotH, double M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litEiii(
// CHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litIfEEKNS_8__detail9enable_ifIXaasr8__detail13is_arithmeticIT_EE5Valuesr8__detail7is_sameIfS3_EE5valueEDv4_S3_E4TypeES3_S3_S3_(
float4 test_lit_int(int NDotL, int NDotH, int M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litEjjj(
// CHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litIfEEKNS_8__detail9enable_ifIXaasr8__detail13is_arithmeticIT_EE5Valuesr8__detail7is_sameIfS3_EE5valueEDv4_S3_E4TypeES3_S3_S3_(
float4 test_lit_uint(uint NDotL, uint NDotH, uint M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litElll(
// CHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litIfEEKNS_8__detail9enable_ifIXaasr8__detail13is_arithmeticIT_EE5Valuesr8__detail7is_sameIfS3_EE5valueEDv4_S3_E4TypeES3_S3_S3_(
float4 test_lit_int64_t(int64_t NDotL, int64_t NDotH, int64_t M) { return lit(NDotL, NDotH, M); }

// CHECK-LABEL: define linkonce_odr noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litEmmm(
// CHECK: call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float> @_ZN4hlsl3litIfEEKNS_8__detail9enable_ifIXaasr8__detail13is_arithmeticIT_EE5Valuesr8__detail7is_sameIfS3_EE5valueEDv4_S3_E4TypeES3_S3_S3_(
float4 test_lit_uint64_t(uint64_t NDotL, uint64_t NDotH, uint64_t M) { return lit(NDotL, NDotH, M); }
