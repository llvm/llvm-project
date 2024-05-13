// RUN: %clang_cc1 -x objective-c -triple=x86_64-apple-darwin9 -emit-llvm -o - < %s | FileCheck %s

// CHECK-DAG: @[[complex_int:.*]] = private unnamed_addr constant [3 x i8] c"ji\00", align 1
// CHECK-DAG: @a ={{.*}} global ptr @[[complex_int]], align 8
char *a = @encode(_Complex int);

// CHECK-DAG: @[[complex_float:.*]] = private unnamed_addr constant [3 x i8] c"jf\00", align 1
// CHECK-DAG: @b ={{.*}} global ptr @[[complex_float]], align 8
char *b = @encode(_Complex float);

// CHECK-DAG: @[[complex_double:.*]] = private unnamed_addr constant [3 x i8] c"jd\00", align 1
// CHECK-DAG: @c ={{.*}} global ptr @[[complex_double]], align 8
char *c = @encode(_Complex double);

// CHECK-DAG: @[[int_128:.*]] = private unnamed_addr constant [2 x i8] c"t\00", align 1
// CHECK-DAG: @e ={{.*}} global ptr @[[int_128]], align 8
char *e = @encode(__int128_t);

// CHECK-DAG: @[[uint_128:.*]] = private unnamed_addr constant [2 x i8] c"T\00", align 1
// CHECK-DAG: @f ={{.*}} global ptr @[[uint_128]], align 8
char *f = @encode(__uint128_t);
