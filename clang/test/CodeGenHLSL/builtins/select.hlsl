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
// CHECK-LABEL: test_select_infer_struct
// CHECK: [[TRUE_VAL:%.*]] = load %struct.S, ptr {{%.*}}, align 1
// CHECK: [[FALSE_VAL:%.*]] = load %struct.S, ptr {{%.*}}, align 1
// CHECK: [[SELECT:%.*]] = select i1 {{%.*}}, %struct.S [[TRUE_VAL]], %struct.S [[FALSE_VAL]]
// CHECK: store %struct.S [[SELECT]], ptr {{%.*}}, align 1
// CHECK: ret void
struct S test_select_infer_struct(bool cond0, struct S tVal, struct S fVal) {
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
  return select(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_2
// CHECK: [[SELECT:%.*]] = select <2 x i1> {{%.*}}, <2 x i32> {{%.*}}, <2 x i32> {{%.*}}
// CHECK: ret <2 x i32> [[SELECT]]
int2 test_select_vector_2(bool2 cond0, int2 tVals, int2 fVals) {
  return select(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_3
// CHECK: [[SELECT:%.*]] = select <3 x i1> {{%.*}}, <3 x i32> {{%.*}}, <3 x i32> {{%.*}}
// CHECK: ret <3 x i32> [[SELECT]]
int3 test_select_vector_3(bool3 cond0, int3 tVals, int3 fVals) {
  return select(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_4
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_vector_4(bool4 cond0, int4 tVals, int4 fVals) {
  return select(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_scalar_vector
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC1]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i32> [[SPLAT1]], <4 x i32> {{%.*}}
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_vector_scalar_vector(bool4 cond0, int tVal, int4 fVals) {
  return select(cond0, tVal, fVals);
}

// CHECK-LABEL: test_select_vector_vector_scalar
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC1]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i32> {{%.*}}, <4 x i32> [[SPLAT1]]
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_vector_vector_scalar(bool4 cond0, int4 tVals, int fVal) {
  return select(cond0, tVals, fVal);
}

// CHECK-LABEL: test_select_vector_scalar_scalar
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC1]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SPLAT_SRC2:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT2:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC2]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> {{%.*}}, <4 x i32> [[SPLAT1]], <4 x i32> [[SPLAT2]]
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_vector_scalar_scalar(bool4 cond0, int tVal, int fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_vector_17
// CHECK: [[SELECT:%.*]] = select <17 x i1> {{%.*}}, <17 x i32> {{%.*}}, <17 x i32> {{%.*}}
// CHECK: ret <17 x i32> [[SELECT]]
vector<int, 17> test_select_vector_17(vector<bool, 17> cond0,
                                   vector<int, 17> tVals,
                                   vector<int, 17> fVals) {
  return select(cond0, tVals, fVals);
}

// CHECK-LABEL: test_select_vector_5_scalar_vector
// CHECK: [[COND:%.*]] = load <5 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <5 x i32> [[COND]], zeroinitializer
// CHECK: [[SELECT:%.*]] = select <5 x i1> [[TOBOOL]], <5 x i32> {{%.*}}, <5 x i32> {{%.*}}
// CHECK: ret <5 x i32> [[SELECT]]
vector<int, 5> test_select_vector_5_scalar_vector(vector<int, 5> cond0,
                                                 int tVal,
                                                 vector<int, 5> fVals) {
  return select(cond0, tVal, fVals);
}

// CHECK-LABEL: test_select_vector_20_vector_scalar
// CHECK: [[SELECT:%.*]] = select <20 x i1> {{%.*}}, <20 x i32> {{%.*}}, <20 x i32> {{%.*}}
// CHECK: ret <20 x i32> [[SELECT]]
vector<int, 20> test_select_vector_20_vector_scalar(vector<bool, 20> cond0,
                                                 vector<int, 20> tVals,
                                                 int fVal) {
  return select(cond0, tVals, fVal);
}

// CHECK-LABEL: test_select_vector_8_scalar_scalar
// CHECK: [[COND:%.*]] = load <8 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <8 x i32> [[COND]], zeroinitializer
// CHECK: [[SELECT:%.*]] = select <8 x i1> [[TOBOOL]], <8 x i32> {{%.*}}, <8 x i32> {{%.*}}
// CHECK: ret <8 x i32> [[SELECT]]
vector<int, 8> test_select_vector_8_scalar_scalar(vector<int, 8> cond0,
                                                 int tVal, int fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_nonbool_cond_vector_4
// CHECK: [[TMP0:%.*]] = load <4 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <4 x i32> [[TMP0]], zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> [[TOBOOL]], <4 x i1> {{%.*}}, <4 x i1> {{%.*}}
// CHECK: ret <4 x i1> [[SELECT]]
bool4 test_select_nonbool_cond_vector_4(int4 cond0, bool4 tVal, bool4 fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_nonbool_cond_vector_scalar_vector
// CHECK: [[TMP0:%.*]] = load <3 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <3 x i32> [[TMP0]], zeroinitializer
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <3 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <3 x i32> [[SPLAT_SRC1]], <3 x i32> poison, <3 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <3 x i1> [[TOBOOL]], <3 x i32> [[SPLAT1]], <3 x i32> {{%.*}}
// CHECK: ret <3 x i32> [[SELECT]]
int3 test_select_nonbool_cond_vector_scalar_vector(int3 cond0, int tVal, int3 fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_nonbool_cond_vector_vector_scalar
// CHECK: [[TMP0:%.*]] = load <2 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <2 x i32> [[TMP0]], zeroinitializer
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <2 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <2 x i32> [[SPLAT_SRC1]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <2 x i1> [[TOBOOL]], <2 x i32> {{%.*}}, <2 x i32> [[SPLAT1]]
// CHECK: ret <2 x i32> [[SELECT]]
int2 test_select_nonbool_cond_vector_vector_scalar(int2 cond0, int2 tVal, int fVal) {
  return select(cond0, tVal, fVal);
}

// CHECK-LABEL: test_select_nonbool_cond_vector_scalar_scalar
// CHECK: [[TMP0:%.*]] = load <4 x i32>, ptr %cond0.addr, align 4
// CHECK: [[TOBOOL:%.*]] = icmp ne <4 x i32> [[TMP0]], zeroinitializer
// CHECK: [[SPLAT_SRC1:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT1:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC1]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SPLAT_SRC2:%.*]] = insertelement <4 x i32> poison, i32 {{%.*}}, i64 0
// CHECK: [[SPLAT2:%.*]] = shufflevector <4 x i32> [[SPLAT_SRC2]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[SELECT:%.*]] = select <4 x i1> [[TOBOOL]], <4 x i32> [[SPLAT1]], <4 x i32> [[SPLAT2]]
// CHECK: ret <4 x i32> [[SELECT]]
int4 test_select_nonbool_cond_vector_scalar_scalar(int4 cond0, int tVal, int fVal) {
  return select(cond0, tVal, fVal);
}
