// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK-LABEL: f3_to_d4
// CHECK: [[f3:%.*]] = alloca <3 x float>
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: store <3 x float> splat (float 1.000000e+00), ptr [[f3]]
// CHECK: [[vecf3:%.*]] = load <3 x float>, ptr [[f3]]
// CHECK: [[vecf4:%.*]] = shufflevector <3 x float> [[vecf3]], <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 0>
// CHECK: [[vecd4:%.*]] = fpext reassoc nnan ninf nsz arcp afn <4 x float> [[vecf4]] to <4 x double>
// CHECK: store <4 x double> [[vecd4]], ptr [[d4]]
void f3_to_d4() {
  vector<float,3> f3 = 1.0;
  vector<double,4> d4 = f3.xyzx;
}

// CHECK-LABEL: f3_to_f2
// CHECK: [[f3:%.*]] = alloca <3 x float>
// CHECK: [[f2:%.*]] = alloca <2 x float>
// CHECK: store <3 x float> splat (float 2.000000e+00), ptr [[f3]]
// CHECK: [[vecf3:%.*]] = load <3 x float>, ptr [[f3]]
// CHECK: [[vecf2:%.*]] = shufflevector <3 x float> [[vecf3]], <3 x float> poison, <2 x i32> <i32 0, i32 1>
// CHECK: store <2 x float> [[vecf2]], ptr [[f2]]
void f3_to_f2() {
  vector<float,3> f3 = 2.0;
  vector<float,2> f2 = f3;
}

// CHECK-LABEL: d4_to_f2
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: [[f2:%.*]] = alloca <2 x float>
// CHECK: store <4 x double> splat (double 3.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[vecf4:%.*]] = fptrunc reassoc nnan ninf nsz arcp afn <4 x double> [[vecd4]] to <4 x float>
// CHECK: [[vecf2:%.*]] = shufflevector <4 x float> [[vecf4]], <4 x float> poison, <2 x i32> <i32 0, i32 1>
// CHECK: store <2 x float> [[vecf2]], ptr [[f2]]
void d4_to_f2() {
  vector<double,4> d4 = 3.0;
  vector<float,2> f2 = d4;
}

// CHECK-LABEL: f2_to_i2
// CHECK: [[f2:%.*]] = alloca <2 x float>
// CHECK: [[i2:%.*]] = alloca <2 x i32>
// CHECK: store <2 x float> splat (float 4.000000e+00), ptr [[f2]]
// CHECK: [[vecf2:%.*]] = load <2 x float>, ptr [[f2]]
// CHECK: [[veci2:%.*]] = fptosi <2 x float> [[vecf2]] to <2 x i32>
// CHECK: store <2 x i32> [[veci2]], ptr [[i2]]
void f2_to_i2() {
  vector<float,2> f2 = 4.0;
  vector<int,2> i2 = f2;
}

// CHECK-LABEL: d4_to_i2
// CHECK: [[f4:%.*]] = alloca <4 x double>
// CHECK: [[i2:%.*]] = alloca <2 x i32>
// CHECK: store <4 x double> splat (double 5.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[veci4:%.*]] = fptosi <4 x double> [[vecd4]] to <4 x i32>
// CHECK: [[veci2:%.*]] = shufflevector <4 x i32> [[veci4]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// CHECK: store <2 x i32> [[veci2]], ptr [[i2]]
void d4_to_i2() {
  vector<double,4> d4 = 5.0;
  vector<int,2> i2 = d4;
}

// CHECK-LABEL: d4_to_l4
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: [[l4:%.*]] = alloca <4 x i64>
// CHECK: store <4 x double> splat (double 6.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[vecl4:%.*]] = fptosi <4 x double> [[vecd4]] to <4 x i64>
// CHECK: store <4 x i64> [[vecl4]], ptr [[l4]]
void d4_to_l4() {
  vector<double,4> d4 = 6.0;
  vector<long,4> l4 = d4;
}


// CHECK-LABEL: l4_to_i2
// CHECK: [[l4:%.*]] = alloca <4 x i64>
// CHECK: [[i2:%.*]] = alloca <2 x i32>
// CHECK: store <4 x i64> splat (i64 7), ptr [[l4]]
// CHECK: [[vecl4:%.*]] = load <4 x i64>, ptr [[l4]]
// CHECK: [[veci4:%.*]] = trunc <4 x i64> [[vecl4]] to <4 x i32>
// CHECK: [[veci2:%.*]] = shufflevector <4 x i32> [[veci4]], <4 x i32> poison, <2 x i32> <i32 0, i32 1>
// CHECK: store <2 x i32> [[veci2]], ptr [[i2]]
void l4_to_i2() {
  vector<long, 4> l4 = 7;
  vector<int,2> i2 = l4;
}

// CHECK-LABEL: i2_to_b2
// CHECK: [[l2:%.*]] = alloca <2 x i32>
// CHECK: [[b2:%.*]] = alloca i8
// CHECK: store <2 x i32> splat (i32 8), ptr [[i2]]
// CHECK: [[veci2:%.*]] = load <2 x i32>, ptr [[i2]]
// CHECK: [[vecb2:%.*]] = icmp ne <2 x i32> [[veci2]], zeroinitializer
// CHECK: [[vecb8:%.*]] = shufflevector <2 x i1> [[vecb2]], <2 x i1> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
// CHECK: [[i8:%.*]] = bitcast <8 x i1> [[vecb8]] to i8
// CHECK: store i8 [[i8]], ptr [[b2]]
void i2_to_b2() {
  vector<int, 2> i2 = 8;
  vector<bool, 2> b2 = i2;
}

// CHECK-LABEL: d4_to_b2
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: [[b2:%.*]] = alloca i8
// CHECK: store <4 x double> splat (double 9.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[vecb4:%.*]] = fcmp reassoc nnan ninf nsz arcp afn une <4 x double> [[vecd4]], zeroinitializer
// CHECK: [[vecd2:%.*]] = shufflevector <4 x i1> [[vecb4]], <4 x i1> poison, <2 x i32> <i32 0, i32 1>
// CHECK: [[vecb8:%.*]] = shufflevector <2 x i1> [[vecd2]], <2 x i1> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
// CHECK: [[i8:%.*]] = bitcast <8 x i1> [[vecb8]] to i8
// CHECK: store i8 [[i8]], ptr [[b2]]
void d4_to_b2() {
  vector<double,4> d4 = 9.0;
  vector<bool, 2> b2 = d4;
}

// CHECK-LABEL: d4_to_d1
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: [[d1:%.*]] = alloca <1 x double>
// CHECK: store <4 x double> splat (double 9.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[vecd1:%.*]] = shufflevector <4 x double> [[vecd4]], <4 x double> poison, <1 x i32> zeroinitializer
// CHECK: store <1 x double> [[vecd1]], ptr [[d1:%.*]], align 8
void d4_to_d1() {
  vector<double,4> d4 = 9.0;
  vector<double,1> d1 = d4;
}

// CHECK-LABEL: d4_to_dScalar
// CHECK: [[d4:%.*]] = alloca <4 x double>
// CHECK: [[d:%.*]] = alloca double
// CHECK: store <4 x double> splat (double 9.000000e+00), ptr [[d4]]
// CHECK: [[vecd4:%.*]] = load <4 x double>, ptr [[d4]]
// CHECK: [[d4x:%.*]] = extractelement <4 x double> [[vecd4]], i32 0
// CHECK: store double [[d4x]], ptr [[d]]
void d4_to_dScalar() {
  vector<double,4> d4 = 9.0;
  double d = d4;
}
