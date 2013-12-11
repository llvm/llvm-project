// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 %s -emit-llvm -o - | FileCheck %s

// PR10878

struct S { S(); S(int); ~S(); int n; };

void *p = new S[2][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
// CHECK: %[[ALLOC:.*]] = call noalias i8* @_Znam(i64 32)
// CHECK: %[[COOKIE:.*]] = bitcast i8* %[[ALLOC]] to i64*
// CHECK: store i64 6, i64* %[[COOKIE]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8* %[[ALLOC]], i64 8
// CHECK: %[[START_AS_S:.*]] = bitcast i8* %[[START_AS_i8]] to %[[S:.*]]*
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
// CHECK: %[[S_0:.*]] = bitcast %[[S]]* %[[START_AS_S]] to [3 x %[[S]]]*
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S]]]* %[[S_0]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_0]], i32 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]]* %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_1]], i32 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]]* %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_2]], i32 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr [3 x %[[S]]]* %[[S_0]], i32 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]]* %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_0]], i32 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]]* %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_1]], i32 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]]* %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_2]], i32 6)
//
// CHECK-NOT: br i1
// CHECK-NOT: call
// CHECK: }

int n;
void *q = new S[n][3]{ { 1, 2, 3 }, { 4, 5, 6 } };

// CHECK-LABEL: define
//
// CHECK: load i32* @n
// CHECK: call {{.*}} @llvm.umul.with.overflow.i64(i64 %[[N:.*]], i64 12)
// CHECK: %[[ELTS:.*]] = mul i64 %[[N]], 3
// CHECK: call {{.*}} @llvm.uadd.with.overflow.i64(i64 %{{.*}}, i64 8)
// CHECK: %[[ALLOC:.*]] = call noalias i8* @_Znam(i64 %{{.*}})
//
// CHECK: %[[COOKIE:.*]] = bitcast i8* %[[ALLOC]] to i64*
// CHECK: store i64 %[[ELTS]], i64* %[[COOKIE]]
// CHECK: %[[START_AS_i8:.*]] = getelementptr inbounds i8* %[[ALLOC]], i64 8
// CHECK: %[[START_AS_S:.*]] = bitcast i8* %[[START_AS_i8]] to %[[S]]*
// CHECK: %[[END_AS_S:.*]] = getelementptr inbounds %[[S]]* %[[START_AS_S]], i64 %[[ELTS]]
//
// Explicit initializers:
//
// { 1, 2, 3 }
//
// CHECK: %[[S_0:.*]] = bitcast %[[S]]* %[[START_AS_S]] to [3 x %[[S]]]*
//
// CHECK: %[[S_0_0:.*]] = getelementptr inbounds [3 x %[[S]]]* %[[S_0]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_0]], i32 1)
// CHECK: %[[S_0_1:.*]] = getelementptr inbounds %[[S]]* %[[S_0_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_1]], i32 2)
// CHECK: %[[S_0_2:.*]] = getelementptr inbounds %[[S]]* %[[S_0_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_0_2]], i32 3)
//
// { 4, 5, 6 }
//
// CHECK: %[[S_1:.*]] = getelementptr [3 x %[[S]]]* %[[S_0]], i32 1
//
// CHECK: %[[S_1_0:.*]] = getelementptr inbounds [3 x %[[S]]]* %[[S_1]], i64 0, i64 0
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_0]], i32 4)
// CHECK: %[[S_1_1:.*]] = getelementptr inbounds %[[S]]* %[[S_1_0]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_1]], i32 5)
// CHECK: %[[S_1_2:.*]] = getelementptr inbounds %[[S]]* %[[S_1_1]], i64 1
// CHECK: call void @_ZN1SC1Ei(%[[S]]* %[[S_1_2]], i32 6)
//
// CHECK: %[[S_2:.*]] = getelementptr [3 x %[[S]]]* %[[S_1]], i32 1
// CHECK: %[[S_2_AS_S:.*]] = bitcast [3 x %[[S]]]* %[[S_2]] to %[[S]]*
// CHECK: icmp eq %[[S]]* %[[S_2_AS_S]], %[[END_AS_S]]
// CHECK: br i1
//
// S[n-2][3] initialization loop:
//
// CHECK: %[[END_INNER:.*]] = getelementptr inbounds %[[S]]* %{{.*}}, i64 3
// CHECK: br label
//
//   S[3] initialization loop:
//
//   CHECK: call void @_ZN1SC1Ev(%[[S]]*
//   CHECK: %[[NEXT_INNER:.*]] = getelementptr inbounds %[[S]]* %{{.*}}, i64 1
//   CHECK: icmp eq %[[S]]* %[[NEXT_INNER]], %[[END_INNER]]
//   CHECK: br i1
//
// CHECK: %[[NEXT_OUTER:.*]] = getelementptr %[[S]]* %{{.*}}, i32 1
// CHECK: icmp eq %[[S]]* %[[NEXT_OUTER]], %[[END_AS_S]]
// CHECK: br i1
//
// CHECK: }
