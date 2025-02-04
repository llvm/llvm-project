// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -emit-pch -o %t %S/cxx_paren_init.h
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -include-pch %t %s -emit-llvm -o - | FileCheck %s

// CHECK-DAG: [[STRUCT_S:%.*]] = type { i32, i32 }
// CHECK-DAG: @{{.*s.*}} = {{(dso_local )?}}global [[STRUCT_S]] { i32 1, i32 2 }, align 4
S s = foo(1, 2);

// CHECK-DAG: [[UNION_U:%.*]] = type { i32 }
// CHECK-DAG: @{{.*u.*}} = {{(dso_local )?}}global [[UNION_U]] { i32 3 }, align 4
U u = baz(3);

// CHECK: define dso_local void @{{.*bar.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[I_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[J_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[ARR:%.*]] = alloca [4 x i32], align 16
// CHECK-NEXT: store i32 [[A:%.*]], ptr [[I_ADDR]], align 4
// CHECK-NEXT: store i32 [[B:%.*]], ptr [[J_ADDR]], align 4
// CHECK-NEXT: [[TMP_0:%.*]] = load i32, ptr [[I_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_0]], ptr [[ARR]], align 4
// CHECK-NEXT: [[ARRINIT_ELEM:%.*]] = getelementptr inbounds i32, ptr [[ARR]], i64 1
// CHECK-NEXT: [[TMP_1:%.*]] = load i32, ptr [[J_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_1]], ptr [[ARRINIT_ELEM]], align 4
// CHECK-NEXT: [[ARRINIT_START:%.*]] = getelementptr inbounds i32, ptr [[ARR]], i64 2
// CHECK-NEXT: [[ARRINIT_END:%.*]] = getelementptr inbounds i32, ptr [[ARR]], i64 4
// CHECK-NEXT: br label [[ARRINIT_BODY:%.*]]
// CHECK: [[ARRINIT_CUR:%.*]] = phi ptr [ [[ARRINIT_START]], %entry ], [ [[ARRINIT_NEXT:%.*]], [[ARRINIT_BODY]] ]
// CHECK-NEXT: store i32 0, ptr [[ARRINIT_CUR]], align 4
// CHECK-NEXT: [[ARRINIT_NEXT]] = getelementptr inbounds i32, ptr [[ARRINIT_CUR]], i64 1
// CHECK-NEXT: [[ARRINIT_DONE:%.*]] = icmp eq ptr [[ARRINIT_NEXT]], [[ARRINIT_END:%.*]]
// CHECK-NEXT: br i1 [[ARRINIT_DONE]], label [[ARRINIT_END1:%.*]], label [[ARRINIT_BODY]]
// CHECK: ret void
