// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "Inputs/stdio.h"
#include <stdint.h>

// CHECK-DAG: @[[STR_0:.*]] = private unnamed_addr constant [3 x i8] c"%s\00",
// CHECK-DAG: @[[STR_1:.*]] = private unnamed_addr constant [9 x i8] c"struct A\00",
// CHECK-DAG: @[[STR_2:.*]] = private unnamed_addr constant [4 x i8] c" {\0A\00",
// CHECK-DAG: @[[STR_4:.*]] = private unnamed_addr constant [3 x i8] c"  \00",
// CHECK-DAG: @[[STR_5:.*]] = private unnamed_addr constant [5 x i8] c"char\00",
// CHECK-DAG: @[[STR_6:.*]] = private unnamed_addr constant [3 x i8] c"i1\00",
// CHECK-DAG: @[[STR_7:.*]] = private unnamed_addr constant [16 x i8] c"%s%s %s = %hhd\0A\00",
// CHECK-DAG: @[[STR_8:.*]] = private unnamed_addr constant [12 x i8] c"signed char\00",
// CHECK-DAG: @[[STR_9:.*]] = private unnamed_addr constant [3 x i8] c"i2\00",
// CHECK-DAG: @[[STR_10:.*]] = private unnamed_addr constant [16 x i8] c"%s%s %s = %hhu\0A\00",
// CHECK-DAG: @[[STR_11:.*]] = private unnamed_addr constant [14 x i8] c"unsigned char\00",
// CHECK-DAG: @[[STR_12:.*]] = private unnamed_addr constant [3 x i8] c"i3\00",
// CHECK-DAG: @[[STR_13:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = %hd\0A\00",
// CHECK-DAG: @[[STR_14:.*]] = private unnamed_addr constant [6 x i8] c"short\00",
// CHECK-DAG: @[[STR_15:.*]] = private unnamed_addr constant [3 x i8] c"i4\00",
// CHECK-DAG: @[[STR_16:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = %hu\0A\00",
// CHECK-DAG: @[[STR_17:.*]] = private unnamed_addr constant [15 x i8] c"unsigned short\00",
// CHECK-DAG: @[[STR_18:.*]] = private unnamed_addr constant [3 x i8] c"i5\00",
// CHECK-DAG: @[[STR_19:.*]] = private unnamed_addr constant [14 x i8] c"%s%s %s = %d\0A\00",
// CHECK-DAG: @[[STR_20:.*]] = private unnamed_addr constant [4 x i8] c"int\00",
// CHECK-DAG: @[[STR_21:.*]] = private unnamed_addr constant [3 x i8] c"i6\00",
// CHECK-DAG: @[[STR_22:.*]] = private unnamed_addr constant [14 x i8] c"%s%s %s = %u\0A\00",
// CHECK-DAG: @[[STR_23:.*]] = private unnamed_addr constant [13 x i8] c"unsigned int\00",
// CHECK-DAG: @[[STR_24:.*]] = private unnamed_addr constant [3 x i8] c"i7\00",
// CHECK-DAG: @[[STR_25:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = %ld\0A\00",
// CHECK-DAG: @[[STR_26:.*]] = private unnamed_addr constant [5 x i8] c"long\00",
// CHECK-DAG: @[[STR_27:.*]] = private unnamed_addr constant [3 x i8] c"i8\00",
// CHECK-DAG: @[[STR_28:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = %lu\0A\00",
// CHECK-DAG: @[[STR_29:.*]] = private unnamed_addr constant [14 x i8] c"unsigned long\00",
// CHECK-DAG: @[[STR_30:.*]] = private unnamed_addr constant [3 x i8] c"i9\00",
// CHECK-DAG: @[[STR_31:.*]] = private unnamed_addr constant [16 x i8] c"%s%s %s = %lld\0A\00",
// CHECK-DAG: @[[STR_32:.*]] = private unnamed_addr constant [10 x i8] c"long long\00",
// CHECK-DAG: @[[STR_33:.*]] = private unnamed_addr constant [4 x i8] c"i10\00",
// CHECK-DAG: @[[STR_34:.*]] = private unnamed_addr constant [16 x i8] c"%s%s %s = %llu\0A\00",
// CHECK-DAG: @[[STR_35:.*]] = private unnamed_addr constant [19 x i8] c"unsigned long long\00",
// CHECK-DAG: @[[STR_36:.*]] = private unnamed_addr constant [4 x i8] c"i11\00",
// CHECK-DAG: @[[STR_37:.*]] = private unnamed_addr constant [14 x i8] c"%s%s %s = %f\0A\00",
// CHECK-DAG: @[[STR_38:.*]] = private unnamed_addr constant [6 x i8] c"float\00",
// CHECK-DAG: @[[STR_39:.*]] = private unnamed_addr constant [3 x i8] c"f1\00",
// CHECK-DAG: @[[STR_40:.*]] = private unnamed_addr constant [7 x i8] c"double\00",
// CHECK-DAG: @[[STR_41:.*]] = private unnamed_addr constant [3 x i8] c"f2\00",
// CHECK-DAG: @[[STR_42:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = %Lf\0A\00",
// CHECK-DAG: @[[STR_43:.*]] = private unnamed_addr constant [12 x i8] c"long double\00",
// CHECK-DAG: @[[STR_44:.*]] = private unnamed_addr constant [3 x i8] c"f3\00",
// CHECK-DAG: @[[STR_45:.*]] = private unnamed_addr constant [14 x i8] c"%s%s %s = %p\0A\00",
// CHECK-DAG: @[[STR_46:.*]] = private unnamed_addr constant [7 x i8] c"void *\00",
// CHECK-DAG: @[[STR_47:.*]] = private unnamed_addr constant [3 x i8] c"p1\00",
// CHECK-DAG: @[[STR_48:.*]] = private unnamed_addr constant [19 x i8] c"%s%s %s = \22%.32s\22\0A\00",
// CHECK-DAG: @[[STR_49:.*]] = private unnamed_addr constant [7 x i8] c"char *\00",
// CHECK-DAG: @[[STR_50:.*]] = private unnamed_addr constant [3 x i8] c"s1\00",
// CHECK-DAG: @[[STR_51:.*]] = private unnamed_addr constant [13 x i8] c"const char *\00",
// CHECK-DAG: @[[STR_52:.*]] = private unnamed_addr constant [3 x i8] c"s2\00",
// CHECK-DAG: @[[STR_53:.*]] = private unnamed_addr constant [15 x i8] c"%s%s %s = *%p\0A\00",
// CHECK-DAG: @[[STR_54:.*]] = private unnamed_addr constant [9 x i8] c"char[10]\00",
// CHECK-DAG: @[[STR_55:.*]] = private unnamed_addr constant [3 x i8] c"s3\00",
// CHECK-DAG: @[[STR_56:.*]] = private unnamed_addr constant [10 x i8] c"%s%s %s =\00",
// CHECK-DAG: @[[STR_57:.*]] = private unnamed_addr constant [9 x i8] c"struct X\00",
// CHECK-DAG: @[[STR_58:.*]] = private unnamed_addr constant [3 x i8] c"x1\00",
// CHECK-DAG: @[[STR_59:.*]] = private unnamed_addr constant [5 x i8] c"    \00",
// CHECK-DAG: @[[STR_60:.*]] = private unnamed_addr constant [2 x i8] c"n\00",
// CHECK-DAG: @[[STR_61:.*]] = private unnamed_addr constant [5 x i8] c"%s}\0A\00",
// CHECK-DAG: @[[STR_62:.*]] = private unnamed_addr constant [3 x i8] c"n1\00",
// CHECK-DAG: @[[STR_63:.*]] = private unnamed_addr constant [3 x i8] c"n2\00",
// CHECK-DAG: @[[STR_64:.*]] = private unnamed_addr constant [3 x i8] c"u1\00",
// CHECK-DAG: @[[STR_65:.*]] = private unnamed_addr constant [3 x i8] c"u2\00",
// CHECK-DAG: @[[STR_66:.*]] = private unnamed_addr constant [20 x i8] c"%s%s %s : %zu = %d\0A\00",
// CHECK-DAG: @[[STR_67:.*]] = private unnamed_addr constant [3 x i8] c"b1\00",
// CHECK-DAG: @[[STR_68:.*]] = private unnamed_addr constant [3 x i8] c"b2\00",
// CHECK-DAG: @[[STR_69:.*]] = private unnamed_addr constant [13 x i8] c"_Complex int\00",
// CHECK-DAG: @[[STR_70:.*]] = private unnamed_addr constant [4 x i8] c"ci1\00",
// CHECK-DAG: @[[STR_71:.*]] = private unnamed_addr constant [16 x i8] c"_Complex double\00",
// CHECK-DAG: @[[STR_72:.*]] = private unnamed_addr constant [4 x i8] c"cd1\00",
// CHECK-DAG: @[[STR_73:.*]] = private unnamed_addr constant [3 x i8] c"}\0A\00",

struct X {
  int n;
};

struct A {
  char i1;
  signed char i2;
  unsigned char i3;
  short i4;
  unsigned short i5;
  int i6;
  unsigned int i7;
  long i8;
  unsigned long i9;
  long long i10;
  unsigned long long i11;

  float f1;
  double f2;
  long double f3;

  void *p1;
  char *s1;
  const char *s2;
  char s3[10];

  struct X x1;

  struct {
    int n1;
    struct X n2;
  };
  union {
    int u1;
    int u2;
  };

  int b1 : 5;
  int : 0;
  int b2 : 3;
  int : 5;

  _Complex int ci1;
  _Complex double cd1;
};

int printf(const char *fmt, ...);

// CHECK-LABEL: define {{.*}} @test(
void test(struct A *a) {
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_0]], ptr noundef @[[STR_1]])

  // CHECK: %[[VAL_0:.*]] = load ptr, ptr %[[VAL_a_addr:.*]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_2]])

  // CHECK: %[[VAL_i1:.*]] = getelementptr inbounds %[[VAL_struct_A:.*]], ptr %[[VAL_0]], i32 0, i32 0
  // CHECK: %[[VAL_1:.*]] = load i8, ptr %[[VAL_i1]],
  // CHECK: %[[VAL_conv:.*]] = sext i8 %[[VAL_1]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_7]], ptr noundef @[[STR_4]], ptr noundef @[[STR_5]], ptr noundef @[[STR_6]], i32 noundef %[[VAL_conv]])

  // CHECK: %[[VAL_i2:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 1
  // CHECK: %[[VAL_2:.*]] = load i8, ptr %[[VAL_i2]],
  // CHECK: %[[VAL_conv3:.*]] = sext i8 %[[VAL_2]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_7]], ptr noundef @[[STR_4]], ptr noundef @[[STR_8]], ptr noundef @[[STR_9]], i32 noundef %[[VAL_conv3]])

  // CHECK: %[[VAL_i3:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 2
  // CHECK: %[[VAL_3:.*]] = load i8, ptr %[[VAL_i3]],
  // CHECK: %[[VAL_conv5:.*]] = zext i8 %[[VAL_3]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_10]], ptr noundef @[[STR_4]], ptr noundef @[[STR_11]], ptr noundef @[[STR_12]], i32 noundef %[[VAL_conv5]])

  // CHECK: %[[VAL_i4:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 3
  // CHECK: %[[VAL_4:.*]] = load i16, ptr %[[VAL_i4]],
  // CHECK: %[[VAL_conv7:.*]] = sext i16 %[[VAL_4]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_13]], ptr noundef @[[STR_4]], ptr noundef @[[STR_14]], ptr noundef @[[STR_15]], i32 noundef %[[VAL_conv7]])

  // CHECK: %[[VAL_i5:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 4
  // CHECK: %[[VAL_5:.*]] = load i16, ptr %[[VAL_i5]],
  // CHECK: %[[VAL_conv9:.*]] = zext i16 %[[VAL_5]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_16]], ptr noundef @[[STR_4]], ptr noundef @[[STR_17]], ptr noundef @[[STR_18]], i32 noundef %[[VAL_conv9]])

  // CHECK: %[[VAL_i6:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 5
  // CHECK: %[[VAL_6:.*]] = load i32, ptr %[[VAL_i6]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_21]], i32 noundef %[[VAL_6]])

  // CHECK: %[[VAL_i7:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 6
  // CHECK: %[[VAL_7:.*]] = load i32, ptr %[[VAL_i7]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_22]], ptr noundef @[[STR_4]], ptr noundef @[[STR_23]], ptr noundef @[[STR_24]], i32 noundef %[[VAL_7]])

  // CHECK: %[[VAL_i8:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 7
  // CHECK: %[[VAL_8:.*]] = load i64, ptr %[[VAL_i8]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_25]], ptr noundef @[[STR_4]], ptr noundef @[[STR_26]], ptr noundef @[[STR_27]], i64 noundef %[[VAL_8]])

  // CHECK: %[[VAL_i9:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 8
  // CHECK: %[[VAL_9:.*]] = load i64, ptr %[[VAL_i9]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_28]], ptr noundef @[[STR_4]], ptr noundef @[[STR_29]], ptr noundef @[[STR_30]], i64 noundef %[[VAL_9]])

  // CHECK: %[[VAL_i10:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 9
  // CHECK: %[[VAL_10:.*]] = load i64, ptr %[[VAL_i10]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_31]], ptr noundef @[[STR_4]], ptr noundef @[[STR_32]], ptr noundef @[[STR_33]], i64 noundef %[[VAL_10]])

  // CHECK: %[[VAL_i11:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 10
  // CHECK: %[[VAL_11:.*]] = load i64, ptr %[[VAL_i11]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_34]], ptr noundef @[[STR_4]], ptr noundef @[[STR_35]], ptr noundef @[[STR_36]], i64 noundef %[[VAL_11]])

  // CHECK: %[[VAL_f1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 11
  // CHECK: %[[VAL_12:.*]] = load float, ptr %[[VAL_f1]],
  // CHECK: %[[VAL_conv17:.*]] = fpext float %[[VAL_12]] to double
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_37]], ptr noundef @[[STR_4]], ptr noundef @[[STR_38]], ptr noundef @[[STR_39]], double noundef %[[VAL_conv17]])

  // CHECK: %[[VAL_f2:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 12
  // CHECK: %[[VAL_13:.*]] = load double, ptr %[[VAL_f2]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_37]], ptr noundef @[[STR_4]], ptr noundef @[[STR_40]], ptr noundef @[[STR_41]], double noundef %[[VAL_13]])

  // CHECK: %[[VAL_f3:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 13
  // CHECK: %[[VAL_14:.*]] = load x86_fp80, ptr %[[VAL_f3]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_42]], ptr noundef @[[STR_4]], ptr noundef @[[STR_43]], ptr noundef @[[STR_44]], x86_fp80 noundef %[[VAL_14]])

  // CHECK: %[[VAL_p1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 14
  // CHECK: %[[VAL_15:.*]] = load ptr, ptr %[[VAL_p1]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_45]], ptr noundef @[[STR_4]], ptr noundef @[[STR_46]], ptr noundef @[[STR_47]], ptr noundef %[[VAL_15]])

  // CHECK: %[[VAL_s1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 15
  // CHECK: %[[VAL_16:.*]] = load ptr, ptr %[[VAL_s1]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_48]], ptr noundef @[[STR_4]], ptr noundef @[[STR_49]], ptr noundef @[[STR_50]], ptr noundef %[[VAL_16]])

  // CHECK: %[[VAL_s2:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 16
  // CHECK: %[[VAL_17:.*]] = load ptr, ptr %[[VAL_s2]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_48]], ptr noundef @[[STR_4]], ptr noundef @[[STR_51]], ptr noundef @[[STR_52]], ptr noundef %[[VAL_17]])

  // CHECK: %[[VAL_s3:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 17
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_53]], ptr noundef @[[STR_4]], ptr noundef @[[STR_54]], ptr noundef @[[STR_55]], ptr noundef %[[VAL_s3]])

  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_56]], ptr noundef @[[STR_4]], ptr noundef @[[STR_57]], ptr noundef @[[STR_58]])

  // CHECK: %[[VAL_x1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 18
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_2]])

  // CHECK: %[[VAL_n:.*]] = getelementptr inbounds %[[VAL_struct_X:.*]], ptr %[[VAL_x1]], i32 0, i32 0
  // CHECK: %[[VAL_18:.*]] = load i32, ptr %[[VAL_n]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_59]], ptr noundef @[[STR_20]], ptr noundef @[[STR_60]], i32 noundef %[[VAL_18]])

  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_61]], ptr noundef @[[STR_4]])

  // CHECK: %[[VAL_19:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 19
  // CHECK: %[[VAL_n1:.*]] = getelementptr inbounds %[[VAL_struct_anon:.*]], ptr %[[VAL_19]], i32 0, i32 0
  // CHECK: %[[VAL_20:.*]] = load i32, ptr %[[VAL_n1]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_62]], i32 noundef %[[VAL_20]])

  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_56]], ptr noundef @[[STR_4]], ptr noundef @[[STR_57]], ptr noundef @[[STR_63]])

  // CHECK: %[[VAL_21:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 19
  // CHECK: %[[VAL_n2:.*]] = getelementptr inbounds %[[VAL_struct_anon]], ptr %[[VAL_21]], i32 0, i32 1
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_2]])

  // CHECK: %[[VAL_n32:.*]] = getelementptr inbounds %[[VAL_struct_X]], ptr %[[VAL_n2]], i32 0, i32 0
  // CHECK: %[[VAL_22:.*]] = load i32, ptr %[[VAL_n32]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_59]], ptr noundef @[[STR_20]], ptr noundef @[[STR_60]], i32 noundef %[[VAL_22]])

  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_61]], ptr noundef @[[STR_4]])

  // CHECK: %[[VAL_23:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 20
  // CHECK: %[[VAL_24:.*]] = load i32, ptr %[[VAL_23]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_64]], i32 noundef %[[VAL_24]])

  // CHECK: %[[VAL_25:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 20
  // CHECK: %[[VAL_26:.*]] = load i32, ptr %[[VAL_25]],
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_19]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_65]], i32 noundef %[[VAL_26]])

  // CHECK: %[[VAL_b1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 21
  // CHECK: %[[VAL_bf_load:.*]] = load i8, ptr %[[VAL_b1]],
  // CHECK: %[[VAL_bf_shl:.*]] = shl i8 %[[VAL_bf_load]], 3
  // CHECK: %[[VAL_bf_ashr:.*]] = ashr i8 %[[VAL_bf_shl]], 3
  // CHECK: %[[VAL_bf_cast:.*]] = sext i8 %[[VAL_bf_ashr]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_66]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_67]], i64 noundef 5, i32 noundef %[[VAL_bf_cast]])

  // CHECK: %[[VAL_b2:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 23
  // CHECK: %[[VAL_bf_load38:.*]] = load i8, ptr %[[VAL_b2]],
  // CHECK: %[[VAL_bf_shl39:.*]] = shl i8 %[[VAL_bf_load38]], 5
  // CHECK: %[[VAL_bf_ashr40:.*]] = ashr i8 %[[VAL_bf_shl39]], 5
  // CHECK: %[[VAL_bf_cast41:.*]] = sext i8 %[[VAL_bf_ashr40]] to i32
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_66]], ptr noundef @[[STR_4]], ptr noundef @[[STR_20]], ptr noundef @[[STR_68]], i64 noundef 3, i32 noundef %[[VAL_bf_cast41]])

  // CHECK: %[[VAL_ci1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 24
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_53]], ptr noundef @[[STR_4]], ptr noundef @[[STR_69]], ptr noundef @[[STR_70]], ptr noundef %[[VAL_ci1]])

  // CHECK: %[[VAL_cd1:.*]] = getelementptr inbounds %[[VAL_struct_A]], ptr %[[VAL_0]], i32 0, i32 25
  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_53]], ptr noundef @[[STR_4]], ptr noundef @[[STR_71]], ptr noundef @[[STR_72]], ptr noundef %[[VAL_cd1]])

  // CHECK: call {{.*}} @printf(ptr noundef @[[STR_73]])
  __builtin_dump_struct(a, printf);

}
