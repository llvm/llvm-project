// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-linux-gnu | FileCheck %s

struct A {
  A(const char *);
};

// CHECK: @arr ={{.*}} global [3 x %struct.S] zeroinitializer
// CHECK: @.str = {{.*}}constant [6 x i8] c"hello\00"
// CHECK: @.str.1 = {{.*}}constant [6 x i8] c"world\00"
// CHECK: @.str.2 = {{.*}}constant [8 x i8] c"goodbye\00"

struct S {
  int n;
  A s;
} arr[] = {
  { 0, "hello" },
  { 1, "world" },
  { 2, "goodbye" }
};

// CHECK: store i32 0, ptr @arr
// CHECK: call void @_ZN1AC1EPKc(ptr {{[^,]*}} getelementptr inbounds nuw (i8, ptr @arr, i64 4), ptr noundef @.str)
// CHECK: store i32 1, ptr getelementptr inbounds nuw (i8, ptr @arr, i64 8)
// CHECK: call void @_ZN1AC1EPKc(ptr {{[^,]*}} getelementptr inbounds nuw (i8, ptr @arr, i64 12), ptr noundef @.str.1)
// CHECK: store i32 2, ptr getelementptr inbounds nuw (i8, ptr @arr, i64 16)
// CHECK: call void @_ZN1AC1EPKc(ptr {{[^,]*}} getelementptr inbounds nuw (i8, ptr @arr, i64 20), ptr noundef @.str.2)
