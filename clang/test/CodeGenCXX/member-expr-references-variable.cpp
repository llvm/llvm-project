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
// CHECK: call void @_Z3useiPKc(i32 noundef 1, ptr noundef @[[STR]])
  Struct s;
  use(2, s.name);
// CHECK: call void @_Z3useiPKc(i32 noundef 2, ptr noundef @[[STR]])
  use(3, ptr->name);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 3, ptr noundef @[[STR]])
  use(4, ref.name);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 4, ptr noundef @[[STR]])
  use(5, Struct(2).name);
// CHECK: call void @_ZN6StructC1Ei(ptr {{[^,]*}} %{{.*}}, i32 noundef 2)
// CHECK: call void @_Z3useiPKc(i32 noundef 5, ptr noundef @[[STR]])
  use(6, getPtr()->name);
// CHECK: call noundef ptr @_Z6getPtrv()
// CHECK: call void @_Z3useiPKc(i32 noundef 6, ptr noundef @[[STR]])
}

void use(int n, __complex float v);

void complexStaticVariableInMemberExpr(Struct *ptr, Struct &ref) {
  use(1, Struct::complexValue);
// CHECK: store float 4.200000e+01, ptr %[[coerce0:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce0]].{{.*}}, align 4
// CHECK: %[[vector0:.*]] = load <2 x float>, ptr %[[coerce0]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 1, <2 x float> noundef %[[vector0]])
  Struct s;
  use(2, s.complexValue);
// CHECK: store float 4.200000e+01, ptr %[[coerce1:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce1]].{{.*}}, align 4
// CHECK: %[[vector1:.*]] = load <2 x float>, ptr %[[coerce1]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 2, <2 x float> noundef %[[vector1]])
  use(3, ptr->complexValue);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: store float 4.200000e+01, ptr %[[coerce2:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce2]].{{.*}}, align 4
// CHECK: %[[vector2:.*]] = load <2 x float>, ptr %[[coerce2]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 3, <2 x float> noundef %[[vector2]])
  use(4, ref.complexValue);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: store float 4.200000e+01, ptr %[[coerce3:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce3]].{{.*}}, align 4
// CHECK: %[[vector3:.*]] = load <2 x float>, ptr %[[coerce3]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 4, <2 x float> noundef %[[vector3]])
  use(5, Struct(2).complexValue);
// CHECK: call void @_ZN6StructC1Ei(ptr {{[^,]*}} %{{.*}}, i32 noundef 2)
// CHECK: store float 4.200000e+01, ptr %[[coerce4:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce4]].{{.*}}, align 4
// CHECK: %[[vector4:.*]] = load <2 x float>, ptr %[[coerce4]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 5, <2 x float> noundef %[[vector4]])
  use(6, getPtr()->complexValue);
// CHECK: call noundef ptr @_Z6getPtrv()
// CHECK: store float 4.200000e+01, ptr %[[coerce5:.*]].{{.*}}, align 4
// CHECK: store float 0.000000e+00, ptr %[[coerce5]].{{.*}}, align 4
// CHECK: %[[vector5:.*]] = load <2 x float>, ptr %[[coerce5]], align 4
// CHECK: call void @_Z3useiCf(i32 noundef 6, <2 x float> noundef %[[vector5]])
}

void aggregateRefInMemberExpr(Struct *ptr, Struct &ref) {
  use(1, Struct::agg.x);
// CHECK: %[[value0:.*]] = load ptr, ptr @_ZGRN6Struct3aggE_, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 1, ptr noundef %[[value0]])
  Struct s;
  use(2, s.agg.x);
// CHECK: %[[value1:.*]] = load ptr, ptr @_ZGRN6Struct3aggE_, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 2, ptr noundef %[[value1]])
  use(3, ptr->agg.x);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: %[[value2:.*]] = load ptr, ptr @_ZGRN6Struct3aggE_, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 3, ptr noundef %[[value2]])
  use(4, ref.agg.x);
// CHECK: load ptr, ptr %{{.*}}, align 8
// CHECK: %[[value3:.*]] = load ptr, ptr @_ZGRN6Struct3aggE_, align 8
// CHECK: call void @_Z3useiPKc(i32 noundef 4, ptr noundef %[[value3]])
}
