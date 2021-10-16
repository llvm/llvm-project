// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct Agg { const char * x; const char * y; constexpr Agg() : x(0), y(0) {} };

struct Struct {
   constexpr static const char *name = "foo";

   constexpr static __complex float complexValue = 42.0;

   static constexpr const Agg &agg = Agg();

   Struct();
   Struct(int x);
};

void use(int n, const char *c);

Struct *getPtr();

// CHECK: @[[STR:.*]] = private unnamed_addr constant [4 x i8] c"foo\00", align 1

void scalarStaticVariableInMemberExpr(Struct *ptr, Struct &ref) {
  use(1, Struct::name);
// CHECK: call void @_Z3useiPKc(i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
  Struct s;
  use(2, s.name);
// CHECK: call void @_Z3useiPKc(i32 noundef 2, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
  use(3, ptr->name);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 3, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
  use(4, ref.name);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 4, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
  use(5, Struct(2).name);
// CHECK: call void @_ZN6StructC1Ei(%struct.Struct* {{[^,]*}} %{{.*}}, i32 noundef 2)
// CHECK: call void @_Z3useiPKc(i32 noundef 5, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
  use(6, getPtr()->name);
// CHECK: call noundef %struct.Struct* @_Z6getPtrv()
// CHECK: call void @_Z3useiPKc(i32 noundef 6, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @[[STR]], i32 0, i32 0))
}

void use(int n, __complex float v);

void complexStaticVariableInMemberExpr(Struct *ptr, Struct &ref) {
  use(1, Struct::complexValue);
// CHECK: store float 4.200000e+01, float* %[[coerce0:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce0]].{{.*}}, align 4
// CHECK: %[[cast0:.*]] = bitcast { float, float }* %[[coerce0]] to <2 x float>*
// CHECK: %[[vector0:.*]] = load <2 x float>, <2 x float>* %[[cast0]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 1, <2 x float> noundef %[[vector0]])
  Struct s;
  use(2, s.complexValue);
// CHECK: store float 4.200000e+01, float* %[[coerce1:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce1]].{{.*}}, align 4
// CHECK: %[[cast1:.*]] = bitcast { float, float }* %[[coerce1]] to <2 x float>*
// CHECK: %[[vector1:.*]] = load <2 x float>, <2 x float>* %[[cast1]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 2, <2 x float> noundef %[[vector1]])
  use(3, ptr->complexValue);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: store float 4.200000e+01, float* %[[coerce2:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce2]].{{.*}}, align 4
// CHECK: %[[cast2:.*]] = bitcast { float, float }* %[[coerce2]] to <2 x float>*
// CHECK: %[[vector2:.*]] = load <2 x float>, <2 x float>* %[[cast2]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 3, <2 x float> noundef %[[vector2]])
  use(4, ref.complexValue);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: store float 4.200000e+01, float* %[[coerce3:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce3]].{{.*}}, align 4
// CHECK: %[[cast3:.*]] = bitcast { float, float }* %[[coerce3]] to <2 x float>*
// CHECK: %[[vector3:.*]] = load <2 x float>, <2 x float>* %[[cast3]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 4, <2 x float> noundef %[[vector3]])
  use(5, Struct(2).complexValue);
// CHECK: call void @_ZN6StructC1Ei(%struct.Struct* {{[^,]*}} %{{.*}}, i32 noundef 2)
// CHECK: store float 4.200000e+01, float* %[[coerce4:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce4]].{{.*}}, align 4
// CHECK: %[[cast4:.*]] = bitcast { float, float }* %[[coerce4]] to <2 x float>*
// CHECK: %[[vector4:.*]] = load <2 x float>, <2 x float>* %[[cast4]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 5, <2 x float> noundef %[[vector4]])
  use(6, getPtr()->complexValue);
// CHECK: call noundef %struct.Struct* @_Z6getPtrv()
// CHECK: store float 4.200000e+01, float* %[[coerce5:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, float* %[[coerce5]].{{.*}}, align 4
// CHECK: %[[cast5:.*]] = bitcast { float, float }* %[[coerce5]] to <2 x float>*
// CHECK: %[[vector5:.*]] = load <2 x float>, <2 x float>* %[[cast5]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 6, <2 x float> noundef %[[vector5]])
}

void aggregateRefInMemberExpr(Struct *ptr, Struct &ref) {
  use(1, Struct::agg.x);
// CHECK: %[[value0:.*]] = load i8*, i8** getelementptr inbounds (%struct.Agg, %struct.Agg* @_ZGRN6Struct3aggE_, i32 0, i32 0), align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 1, i8* noundef %[[value0]])
  Struct s;
  use(2, s.agg.x);
// CHECK: %[[value1:.*]] = load i8*, i8** getelementptr inbounds (%struct.Agg, %struct.Agg* @_ZGRN6Struct3aggE_, i32 0, i32 0), align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 2, i8* noundef %[[value1]])
  use(3, ptr->agg.x);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: %[[value2:.*]] = load i8*, i8** getelementptr inbounds (%struct.Agg, %struct.Agg* @_ZGRN6Struct3aggE_, i32 0, i32 0), align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 3, i8* noundef %[[value2]])
  use(4, ref.agg.x);
// CHECK: load %struct.Struct*, %struct.Struct** %{{.*}}, align 8
// CHECK: %[[value3:.*]] = load i8*, i8** getelementptr inbounds (%struct.Agg, %struct.Agg* @_ZGRN6Struct3aggE_, i32 0, i32 0), align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 4, i8* noundef %[[value3]])
}
