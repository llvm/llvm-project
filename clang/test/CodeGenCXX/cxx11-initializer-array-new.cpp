// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 %s -emit-llvm -o - | FileCheck %s

// PR10878

struct S { S(); S(int); ~S(); int n; };

void *p = new S[2][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 32)
// CHECK: store i64 6, ptr %[[ALLOC]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8, ptr %[[ALLOC]], i64 8
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S:.*]]], ptr %[[START_AS_i8]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_0]], i32 noundef 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]], ptr %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_1]], i32 noundef 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]], ptr %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_2]], i32 noundef 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[START_AS_i8]], i64 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_0]], i32 noundef 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]], ptr %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_1]], i32 noundef 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]], ptr %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_2]], i32 noundef 6)
//
// CHECK-NOT: br i1
// CHECK-NOT: call
// CHECK: }

int n;
void *q = new S[n][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
//
// CHECK: load i32, ptr @n
// CHECK: call {{.*}} @llvm.umul.with.overflow.i64(i64 %[[N:.*]], i64 12)
// CHECK: %[[ELTS:.*]] = mul i64 %[[N]], 3
// CHECK: call {{.*}} @llvm.uadd.with.overflow.i64(i64 %{{.*}}, i64 8)
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef %{{.*}})
//
// CHECK: store i64 %[[ELTS]], ptr %[[ALLOC]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8, ptr %[[ALLOC]], i64 8
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[START_AS_i8]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_0]], i32 noundef 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]], ptr %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_1]], i32 noundef 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]], ptr %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_0_2]], i32 noundef 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[START_AS_i8]], i64 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_0]], i32 noundef 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]], ptr %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_1]], i32 noundef 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]], ptr %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %[[S_1_2]], i32 noundef 6)
//
// And the rest.
//
// CHECK: %[[S_2:.*]] = getelementptr inbounds [3 x %[[S]]], ptr %[[S_1]], i64 1
//
// CHECK: %[[REST:.*]] = sub i64 %[[ELTS]], 6
// CHECK: icmp eq i64 %[[REST]], 0
// CHECK: br i1
//
// CHECK: %[[END:.*]] = getelementptr inbounds %[[S]], ptr %[[S_2]], i64 %[[REST]]
// CHECK: br label
//
// CHECK: %[[CUR:.*]] = phi ptr [ %[[S_2]], {{.*}} ], [ %[[NEXT:.*]], {{.*}} ]
// CHECK: call void @_ZN1SC1Ev(ptr {{[^,]*}} %[[CUR]])
// CHECK: %[[NEXT]] = getelementptr inbounds %[[S]], ptr %[[CUR]], i64 1
// CHECK: icmp eq ptr %[[NEXT]], %[[END]]
// CHECK: br i1
//
// CHECK: }

struct T { int a; };
void *r = new T[n][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
//
// CHECK: load i32, ptr @n
// CHECK: call {{.*}} @llvm.umul.with.overflow.i64(i64 %[[N:.*]], i64 12)
// CHECK: %[[ELTS:.*]] = mul i64 %[[N]], 3
//
// No cookie.
// CHECK-NOT: @llvm.uadd.with.overflow
//
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef %{{.*}})
//
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
//
// CHECK: %[[T_0_0:.*]] = getelementptr inbounds [3 x %[[T:.*]]], ptr %[[ALLOC]], i64 0, i64 0
// CHECK: %[[T_0_0_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_0_0]], i32 0, i32 0
// CHECK: store i32 1, ptr %[[T_0_0_0]]
// CHECK: %[[T_0_1:.*]] = getelementptr inbounds %[[T]], ptr %[[T_0_0]], i64 1
// CHECK: %[[T_0_1_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_0_1]], i32 0, i32 0
// CHECK: store i32 2, ptr %[[T_0_1_0]]
// CHECK: %[[T_0_2:.*]] = getelementptr inbounds %[[T]], ptr %[[T_0_1]], i64 1
// CHECK: %[[T_0_2_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_0_2]], i32 0, i32 0
// CHECK: store i32 3, ptr %[[T_0_2_0]]
//
// { 4, 5, 6 }
//
// CHECK: %[[T_1:.*]] = getelementptr inbounds [3 x %[[T]]], ptr %[[ALLOC]], i64 1
//
// CHECK: %[[T_1_0:.*]] = getelementptr inbounds [3 x %[[T]]], ptr %[[T_1]], i64 0, i64 0
// CHECK: %[[T_1_0_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_1_0]], i32 0, i32 0
// CHECK: store i32 4, ptr %[[T_1_0_0]]
// CHECK: %[[T_1_1:.*]] = getelementptr inbounds %[[T]], ptr %[[T_1_0]], i64 1
// CHECK: %[[T_1_1_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_1_1]], i32 0, i32 0
// CHECK: store i32 5, ptr %[[T_1_1_0]]
// CHECK: %[[T_1_2:.*]] = getelementptr inbounds %[[T]], ptr %[[T_1_1]], i64 1
// CHECK: %[[T_1_2_0:.*]] = getelementptr inbounds %[[T]], ptr %[[T_1_2]], i32 0, i32 0
// CHECK: store i32 6, ptr %[[T_1_2_0]]
//
// And the rest gets memset to 0.
//
// CHECK: %[[T_2:.*]] = getelementptr inbounds [3 x %[[T]]], ptr %[[T_1]], i64 1
//
// CHECK: %[[SIZE:.*]] = sub i64 %{{.*}}, 24
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %[[T_2]], i8 0, i64 %[[SIZE]], i1 false)
//
// CHECK: }
