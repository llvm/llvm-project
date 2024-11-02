// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s

// CHECK-DAG: @[[STR_0:.*]] = {{.*}} [3 x i8] c"%s\00",
// CHECK-DAG: @[[STR_1:.*]] = {{.*}} [2 x i8] c"C\00",
// CHECK-DAG: @[[STR_2:.*]] = {{.*}} [4 x i8] c" {\0A\00",
// CHECK-DAG: @[[STR_3:.*]] = {{.*}} [5 x i8] c"%s%s\00",
// CHECK-DAG: @[[STR_4:.*]] = {{.*}} [3 x i8] c"  \00",
// CHECK-DAG: @[[STR_5:.*]] = {{.*}} [2 x i8] c"A\00",
// CHECK-DAG: @[[STR_6:.*]] = {{.*}} [14 x i8] c"%s%s %s = %d\0A\00",
// CHECK-DAG: @[[STR_7:.*]] = {{.*}} [5 x i8] c"    \00",
// CHECK-DAG: @[[STR_8:.*]] = {{.*}} [4 x i8] c"int\00",
// CHECK-DAG: @[[STR_9:.*]] = {{.*}} [2 x i8] c"n\00",
// CHECK-DAG: @[[STR_10:.*]] = {{.*}} [5 x i8] c"%s}\0A\00",
// CHECK-DAG: @[[STR_11:.*]] = {{.*}} [2 x i8] c"B\00",
// CHECK-DAG: @[[STR_12:.*]] = {{.*}} [10 x i8] c"%s%s %s =\00",
// CHECK-DAG: @[[STR_13:.*]] = {{.*}} [2 x i8] c"a\00",
// CHECK-DAG: @[[STR_14:.*]] = {{.*}} [15 x i8] c"%s%s %s = *%p\0A\00",
// CHECK-DAG: @[[STR_15:.*]] = {{.*}} [2 x i8] c"X\00",
// CHECK-DAG: @[[STR_16:.*]] = {{.*}} [2 x i8] c"x\00",
// CHECK-DAG: @[[STR_17:.*]] = {{.*}} [2 x i8] c"f\00",
// CHECK-DAG: @[[STR_18:.*]] = {{.*}} [2 x i8] c"g\00",
// CHECK-DAG: @[[STR_19:.*]] = {{.*}} [3 x i8] c"}\0A\00",

struct A { int n; };
struct B { int n; };
class X { private: int n; };
struct C : A, B { A a; X x; int f, g; };

template<typename ...T> int format(int a, const char *str, T ...);

int f();

// CHECK-LABEL: define {{.*}} @_Z1gR1C(
void g(C &c) {
  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_0]], ptr noundef @[[STR_1]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 {{.*}}, ptr noundef @[[STR_2]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_EEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_3]], ptr noundef @[[STR_4]], ptr noundef @[[STR_5]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 {{.*}}, ptr noundef @[[STR_2]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_n:.*]] = getelementptr inbounds %[[VAL_struct_A:.*]], ptr %[[VAL_0:.*]], i32 0, i32 0
  // CHECK: %[[VAL_1:.*]] = load i32, ptr %[[VAL_n]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_6]], ptr noundef @[[STR_7]], ptr noundef @[[STR_8]], ptr noundef @[[STR_9]], i32 noundef %[[VAL_1]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_10]], ptr noundef @[[STR_4]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_EEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_3]], ptr noundef @[[STR_4]], ptr noundef @[[STR_11]])

  // CHECK: %[[VAL_2:.*]] = icmp eq ptr %[[VAL_0]], null
  // CHECK: br i1 %[[VAL_2]],

  // CHECK: %[[VAL_add_ptr:.*]] = getelementptr inbounds i8, ptr %[[VAL_0]], i64 4
  // CHECK: br label

  // CHECK: %[[VAL_cast_result:.*]] = phi
  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 {{.*}}, ptr noundef @[[STR_2]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_n17:.*]] = getelementptr inbounds %[[VAL_struct_B:.*]], ptr %[[VAL_cast_result]], i32 0, i32 0
  // CHECK: %[[VAL_3:.*]] = load i32, ptr %[[VAL_n17]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_6]], ptr noundef @[[STR_7]], ptr noundef @[[STR_8]], ptr noundef @[[STR_9]], i32 noundef %[[VAL_3]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_10]], ptr noundef @[[STR_4]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_EEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_12]], ptr noundef @[[STR_4]], ptr noundef @[[STR_5]], ptr noundef @[[STR_13]])

  // CHECK: %[[VAL_a:.*]] = getelementptr inbounds %[[VAL_struct_C:.*]], ptr %[[VAL_0]], i32 0, i32 2
  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 {{.*}}, ptr noundef @[[STR_2]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_n26:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_a]], i32 0, i32 0
  // CHECK: %[[VAL_4:.*]] = load i32, ptr %[[VAL_n26]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_6]], ptr noundef @[[STR_7]], ptr noundef @[[STR_8]], ptr noundef @[[STR_9]], i32 noundef %[[VAL_4]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJPKcEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_10]], ptr noundef @[[STR_4]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_x:.*]] = getelementptr inbounds %[[VAL_struct_C]], ptr %[[VAL_0]], i32 0, i32 3
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_P1XEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_14]], ptr noundef @[[STR_4]], ptr noundef @[[STR_15]], ptr noundef @[[STR_16]], ptr noundef %[[VAL_x]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_f:.*]] = getelementptr inbounds %[[VAL_struct_C]], ptr %[[VAL_0]], i32 0, i32 4
  // CHECK: %[[VAL_5:.*]] = load i32, ptr %[[VAL_f]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_6]], ptr noundef @[[STR_4]], ptr noundef @[[STR_8]], ptr noundef @[[STR_17]], i32 noundef %[[VAL_5]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: %[[VAL_g:.*]] = getelementptr inbounds %[[VAL_struct_C]], ptr %[[VAL_0]], i32 0, i32 5
  // CHECK: %[[VAL_6:.*]] = load i32, ptr %[[VAL_g]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 {{.*}}, ptr noundef @[[STR_6]], ptr noundef @[[STR_4]], ptr noundef @[[STR_8]], ptr noundef @[[STR_18]], i32 noundef %[[VAL_6]])

  // CHECK: call {{.*}} @_Z1fv()
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 {{.*}}, ptr noundef @[[STR_19]])
  __builtin_dump_struct(&c, format, f());
}

// CHECK-LABEL: define {{.*}} @_Z1hR1X(
void h(X &x) {
  // CHECK: %[[VAL_x_addr:.*]] = alloca ptr,
  // CHECK: store ptr %[[VAL_x]], ptr %[[VAL_x_addr]],
  // CHECK: call {{.*}} @_Z6formatIJPKcEEiiS1_DpT_(i32 noundef 0, ptr noundef @[[STR_0]], ptr noundef @[[STR_15]])

  // CHECK: %[[VAL_0:.*]] = load ptr, ptr %[[VAL_x_addr]],
  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 noundef 0, ptr noundef @[[STR_2]])

  // CHECK: %[[VAL_n:.*]] = getelementptr inbounds %[[VAL_class_X:.*]], ptr %[[VAL_0]], i32 0, i32 0
  // CHECK: %[[VAL_1:.*]] = load i32, ptr %[[VAL_n]],
  // CHECK: call {{.*}} @_Z6formatIJPKcS1_S1_iEEiiS1_DpT_(i32 noundef 0, ptr noundef @[[STR_6]], ptr noundef @[[STR_4]], ptr noundef @[[STR_8]], ptr noundef @[[STR_9]], i32 noundef %[[VAL_1]])

  // CHECK: call {{.*}} @_Z6formatIJEEiiPKcDpT_(i32 noundef 0, ptr noundef @[[STR_19]])
  __builtin_dump_struct(&x, format, 0);
}
