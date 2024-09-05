// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK

// CHECK: %hlsl.select = select i1
// CHECK: ret i32 %hlsl.select
int test_select_bool_int(bool cond0, int tVal, int fVal) {
  return select<int>(cond0, tVal, fVal); }

struct S { int a; };
// CHECK: %hlsl.select = select i1
// CHECK: store ptr %hlsl.select
// CHECK: ret void
struct S test_select_infer(bool cond0, struct S tVal, struct S fVal) {
  return select(cond0, tVal, fVal); }

// CHECK: %hlsl.select = select i1
// CHECK: ret <2 x i32> %hlsl.select
int2 test_select_bool_vector(bool cond0, int2 tVal, int2 fVal) {
  return select<int2>(cond0, tVal, fVal); }

// CHECK: %hlsl.select = select <1 x i1>
// CHECK: ret <1 x i32> %hlsl.select
int1 test_select_vector_2(bool1 cond0, int1 tVals, int1 fVals) {
  return select<int,1>(cond0, tVals, fVals); }

// CHECK: %hlsl.select = select <2 x i1>
// CHECK: ret <2 x i32> %hlsl.select
int2 test_select_vector_2(bool2 cond0, int2 tVals, int2 fVals) {
  return select<int,2>(cond0, tVals, fVals); }

// CHECK: %hlsl.select = select <3 x i1>
// CHECK: ret <3 x i32> %hlsl.select
int3 test_select_vector_3(bool3 cond0, int3 tVals, int3 fVals) {
  return select<int,3>(cond0, tVals, fVals); }

// CHECK: %hlsl.select = select <4 x i1>
// CHECK: ret <4 x i32> %hlsl.select
int4 test_select_vector_4(bool4 cond0, int4 tVals, int4 fVals) {
  return select(cond0, tVals, fVals); }

