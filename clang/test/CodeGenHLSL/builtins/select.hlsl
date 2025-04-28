// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: test_select_bool_int
// CHECK: [[SELECT:%.*]] = select i1 {{%.*}}, i32 {{%.*}}, i32 {{%.*}}
// CHECK: ret i32 [[SELECT]]
int test_select_bool_int(bool cond0, int tVal, int fVal) {
  return select<int>(cond0, tVal, fVal);
}

struct S { int a; };
// CHECK-LABEL: test_select_infer
// CHECK: [[SELECT:%.*]] = select i1 {{%.*}}, ptr {{%.*}}, ptr {{%.*}}
// CHECK: store ptr [[SELECT]]
// CHECK: ret void
struct S test_select_infer(bool cond0, struct S tVal, struct S fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_bool_vector
// CHECK: [[SELECT:%.*]] = select i1 {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> {{%.*}}
// CHECK: ret <2 x i32> [[SELECT]]
int2 test_select_bool_vector(bool cond0, int2 tVal, int2 fVal) {
  return select<int2>(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_vector_1
// CHECK: [[SELECT:%.*]] = select <1 x i1> {{%.*}}, <1 x i32> {{%.*}}, <1 x i32> {{%.*}}
// CHECK: ret <1 x i32> [[SELECT]]
int1 test_select_vector_1(bool1 cond0, int1 tVals, int1 fVals) {
  return select<int,1>(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_2
// CHECK: [[SELECT:%.*]] = select <2 x i1> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> {{%.*}}
// CHECK: ret <2 x i32> [[SELECT]]
int2 test_select_vector_2(bool2 cond0, int2 tVals, int2 fVals) {
  return select<int,2>(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_3
// CHECK: [[SELECT:%.*]] = select <3 x i1> {{%.*}}, <3 x i32> {{%.*}}, <3 x i32> {{%.*}}
// CHECK: ret <3 x i32> [[SELECT]]
int3 test_select_vector_3(bool3 cond0, int3 tVals, int3 fVals) {
  return select<int,3>(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_4
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_vector_4(bool4 cond0, int4 tVals, int4 fVals) {
  return select(cond0, tVals, fVals);
}
