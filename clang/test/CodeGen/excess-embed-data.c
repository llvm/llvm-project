// RUN: %clang_cc1 -std=c23 -emit-llvm -triple x86_64 %s -o - | FileCheck %s

struct S {
  int arr[3];
};

struct S1 {
  struct S s;
};

// CHECK: @[[BConst:.*]] = private unnamed_addr constant [2 x i32] [i32 47, i32 47]
// CHECK: @[[DConst:.*]] = private unnamed_addr constant [2 x i8] c"//"
// CHECK: @[[SConst:.*]] = private unnamed_addr constant %struct.S { [3 x i32] [i32 47, i32 47, i32 32] }
// CHECK: @[[S1Const:.*]] = private unnamed_addr constant %struct.S1 { %struct.S { [3 x i32] [i32 47, i32 47, i32 32] } }

void cases(int x) {
  int a[3] = {x, x,
#embed __FILE__
  };

  int b[2] = {
#embed __FILE__
  };

  char d[2] = {
#embed __FILE__
  };

  struct S s = {
#embed __FILE__
  , x
  };

  struct S1 s1 = {
#embed __FILE__
  , x
  };
}
// CHECK: define dso_local void @cases(i32 noundef %[[X:.*]])
// CHECK:  %[[A:.*]] = alloca [3 x i32]
// CHECK:  %[[B:.*]] = alloca [2 x i32]
// CHECK:  %[[D:.*]] = alloca [2 x i8]
// CHECK:  %[[S:.*]] = alloca %struct.S
// CHECK:  %[[S1:.*]] = alloca %struct.S1
// CHECK:  %[[LX:.*]] = load i32, ptr %[[X]].addr
// CHECK:  store i32 %[[LX]], ptr %[[A]]
// CHECK:  %[[GEP1:.*]] = getelementptr inbounds i32, ptr %[[A]], i64 1
// CHECK:  %[[LX1:.*]] = load i32, ptr %[[X]].addr
// CHECK:  store i32 %1, ptr %arrayinit.element
// CHECK:  %[[GEP1:.*]] = getelementptr inbounds i32, ptr %[[A]], i64 2
// CHECK:  store i32 47, ptr %[[GEP1]]
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[B]], ptr align 4 @[[BConst]], i64 8, i1 false)
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[D]], ptr align 1 @[[DConst]], i64 2, i1 false)
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S]], ptr align 4 @[[SConst]], i64 12, i1 false)
// CHECK:  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S1]], ptr align 4 @[[S1Const]], i64 12, i1 false)
