// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-linux-gnu -std=c++11 %s -emit-llvm -o - | FileCheck %s

// PR10878

struct S { S(); S(int); ~S(); int n; };

void *p = new S[2][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull i8* @_Znam(i64 noundef 32)
// CHECK: %[[COOKIE:.*]] = bitcast i8* %[[ALLOC]] to i64*
// CHECK: store i64 6, i64* %[[COOKIE]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8, i8* %[[ALLOC]], i64 8
// CHECK: %[[START_AS_S:.*]] = bitcast i8* %[[START_AS_i8]] to %[[S:.*]]*
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
// CHECK: %[[S_0:.*]] = bitcast %[[S]]* %[[START_AS_S]] to [3 x %[[S]]]*
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_0]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_0]], i32 noundef 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_1]], i32 noundef 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_2]], i32 noundef 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_0]], i64 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_0]], i32 noundef 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_1]], i32 noundef 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_2]], i32 noundef 6)
//
// CHECK-NOT: br i1
// CHECK-NOT: call
// CHECK: }

int n;
void *q = new S[n][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
//
// CHECK: load i32, i32* @n
// CHECK: call {{.*}} @llvm.umul.with.overflow.i64(i64 %[[N:.*]], i64 12)
// CHECK: %[[ELTS:.*]] = mul i64 %[[N]], 3
// CHECK: call {{.*}} @llvm.uadd.with.overflow.i64(i64 %{{.*}}, i64 8)
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull i8* @_Znam(i64 noundef %{{.*}})
//
// CHECK: %[[COOKIE:.*]] = bitcast i8* %[[ALLOC]] to i64*
// CHECK: store i64 %[[ELTS]], i64* %[[COOKIE]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8, i8* %[[ALLOC]], i64 8
// CHECK: %[[START_AS_S:.*]] = bitcast i8* %[[START_AS_i8]] to %[[S]]*
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
// CHECK: %[[S_0:.*]] = bitcast %[[S]]* %[[START_AS_S]] to [3 x %[[S]]]*
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_0]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_0]], i32 noundef 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_1]], i32 noundef 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_0_2]], i32 noundef 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_0]], i64 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_0]], i32 noundef 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_1]], i32 noundef 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* {{[^,]*}} %[[S_1_2]], i32 noundef 6)
//
// And the rest.
//
// CHECK: %[[S_2:.*]] = getelementptr inbounds [3 x %[[S]]], [3 x %[[S]]]* %[[S_1]], i64 1
// CHECK: %[[S_2_AS_S:.*]] = bitcast [3 x %[[S]]]* %[[S_2]] to %[[S]]*
//
// CHECK: %[[REST:.*]] = sub i64 %[[ELTS]], 6
// CHECK: icmp eq i64 %[[REST]], 0
// CHECK: br i1
//
// CHECK: %[[END:.*]] = getelementptr inbounds %[[S]], %[[S]]* %[[S_2_AS_S]], i64 %[[REST]]
// CHECK: br label
//
// CHECK: %[[CUR:.*]] = phi %[[S]]* [ %[[S_2_AS_S]], {{.*}} ], [ %[[NEXT:.*]], {{.*}} ]
// CHECK: call void @_ZN1SC1Ev(%[[S]]* {{[^,]*}} %[[CUR]])
// CHECK: %[[NEXT]] = getelementptr inbounds %[[S]], %[[S]]* %[[CUR]], i64 1
// CHECK: icmp eq %[[S]]* %[[NEXT]], %[[END]]
// CHECK: br i1
//
// CHECK: }

struct T { int a; };
void *r = new T[n][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
//
// CHECK: load i32, i32* @n
// CHECK: call {{.*}} @llvm.umul.with.overflow.i64(i64 %[[N:.*]], i64 12)
// CHECK: %[[ELTS:.*]] = mul i64 %[[N]], 3
//
// No cookie.
// CHECK-NOT: @llvm.uadd.with.overflow
//
// CHECK: %[[ALLOC:.*]] = call noalias noundef nonnull i8* @_Znam(i64 noundef %{{.*}})
//
// CHECK: %[[START_AS_T:.*]] = bitcast i8* %[[ALLOC]] to %[[T:.*]]*
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
// CHECK: %[[T_0:.*]] = bitcast %[[T]]* %[[START_AS_T]] to [3 x %[[T]]]*
//
// CHECK: %[[T_0_0:.*]] = getelementptr inbounds [3 x %[[T]]], [3 x %[[T]]]* %[[T_0]], i64 0, i64 0
// CHECK: %[[T_0_0_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_0_0]], i32 0, i32 0
// CHECK: store i32 1, i32* %[[T_0_0_0]]
// CHECK: %[[T_0_1:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_0_0]], i64 1
// CHECK: %[[T_0_1_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_0_1]], i32 0, i32 0
// CHECK: store i32 2, i32* %[[T_0_1_0]]
// CHECK: %[[T_0_2:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_0_1]], i64 1
// CHECK: %[[T_0_2_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_0_2]], i32 0, i32 0
// CHECK: store i32 3, i32* %[[T_0_2_0]]
//
// { 4, 5, 6 }
//
// CHECK: %[[T_1:.*]] = getelementptr inbounds [3 x %[[T]]], [3 x %[[T]]]* %[[T_0]], i64 1
//
// CHECK: %[[T_1_0:.*]] = getelementptr inbounds [3 x %[[T]]], [3 x %[[T]]]* %[[T_1]], i64 0, i64 0
// CHECK: %[[T_1_0_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_1_0]], i32 0, i32 0
// CHECK: store i32 4, i32* %[[T_1_0_0]]
// CHECK: %[[T_1_1:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_1_0]], i64 1
// CHECK: %[[T_1_1_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_1_1]], i32 0, i32 0
// CHECK: store i32 5, i32* %[[T_1_1_0]]
// CHECK: %[[T_1_2:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_1_1]], i64 1
// CHECK: %[[T_1_2_0:.*]] = getelementptr inbounds %[[T]], %[[T]]* %[[T_1_2]], i32 0, i32 0
// CHECK: store i32 6, i32* %[[T_1_2_0]]
//
// And the rest gets memset to 0.
//
// CHECK: %[[T_2:.*]] = getelementptr inbounds [3 x %[[T]]], [3 x %[[T]]]* %[[T_1]], i64 1
// CHECK: %[[T_2_AS_T:.*]] = bitcast [3 x %[[T]]]* %[[T_2]] to %[[T]]*
//
// CHECK: %[[SIZE:.*]] = sub i64 %{{.*}}, 24
// CHECK: %[[REST:.*]] = bitcast %[[T]]* %[[T_2_AS_T]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 %[[REST]], i8 0, i64 %[[SIZE]], i1 false)
//
// CHECK: }
