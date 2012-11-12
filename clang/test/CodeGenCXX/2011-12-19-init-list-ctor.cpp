// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

struct A {
  A(const char *);
};

// CHECK: @arr = global [3 x %struct.S] zeroinitializer
// CHECK: @.str = {{.*}}constant [6 x i8] c"hello\00"
// CHECK: @.str1 = {{.*}}constant [6 x i8] c"world\00"
// CHECK: @.str2 = {{.*}}constant [8 x i8] c"goodbye\00"

struct S {
  int n;
  A s;
} arr[] = {
  { 0, "hello" },
  { 1, "world" },
  { 2, "goodbye" }
};

// CHECK: store i32 0, i32* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 0, i32 0)
// CHECK: call void @_ZN1AC1EPKc(%struct.A* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 0, i32 1), i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0))
// CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 1, i32 0)
// CHECK: call void @_ZN1AC1EPKc(%struct.A* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 1, i32 1), i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0))
// CHECK: store i32 2, i32* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 2, i32 0)
// CHECK: call void @_ZN1AC1EPKc(%struct.A* getelementptr inbounds ([3 x %struct.S]* @arr, i64 0, i64 2, i32 1), i8* getelementptr inbounds ([8 x i8]* @.str2, i32 0, i32 0))
