// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

int a[1];
// CHECK: @a = global [1 x i32] zeroinitializer
template <int>
void test_transform() {
  auto [b] = a;
}
void (*d)(){test_transform<0>};
// CHECK-LABEL: define {{.*}} @_Z14test_transformILi0EEvv
// CHECK:       [[ENTRY:.*]]:
// CHECK-NEXT:  [[ARR:%.*]] = alloca [1 x i32]
// CHECK-NEXT:  [[BEGIN:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* [[ARR]], i64 0, i64 0
// CHECK-NEXT:  br label %[[BODY:.*]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[BODY]]:
// CHECK-NEXT:  [[CUR:%.*]] = phi i64 [ 0, %[[ENTRY]] ], [ [[NEXT:%.*]], %[[BODY]] ]
// CHECK-NEXT:  [[DEST:%.*]] = getelementptr inbounds i32, i32* [[BEGIN]], i64 [[CUR]]
// CHECK-NEXT:  [[SRC:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* @a, i64 0, i64 [[CUR]]
// CHECK-NEXT:  [[X:%.*]] = load i32, i32* [[SRC]]
// CHECK-NEXT:  store i32 [[X]], i32* [[DEST]]
// CHECK-NEXT:  [[NEXT]] = add nuw i64 [[CUR]], 1
// CHECK-NEXT:  [[EQ:%.*]] = icmp eq i64 [[NEXT]], 1
// CHECK-NEXT:  br i1 [[EQ]], label %[[FIN:.*]], label %[[BODY]]
// CHECK-EMPTY:
// CHECK-NEXT:  [[FIN]]:
// CHECK-NEXT:  ret void
