// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify assume_safety vectorization is recognized.
void vectorize_test(int *List, int Length) {
// CHECK: define {{.*}} @_Z14vectorize_test
// CHECK: [[LOAD1_IV:.+]] = load i32, i32* [[IV1:[^,]+]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_2:[0-9]+]]
// CHECK-NEXT: [[LOAD1_LEN:.+]] = load i32, i32* [[LEN1:.+]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_2]]
// CHECK-NEXT: [[CMP1:.+]] = icmp slt i32[[LOAD1_IV]],[[LOAD1_LEN]]
// CHECK-NEXT: br i1[[CMP1]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: [[RHIV1:.+]] = load i32, i32* [[IV1]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_2]]
    // CHECK-DAG: [[CALC1:.+]] = mul nsw i32[[RHIV1]], 2
    // CHECK-DAG: [[SIV1:.+]] = load i32, i32* [[IV1]]{{.*}}!llvm.access.group ![[ACCESS_GROUP_2]]
    // CHECK-DAG: [[INDEX1:.+]] = sext i32[[SIV1]] to i64
    // CHECK-DAG: [[ARRAY1:.+]] = load i32*, i32** [[LIST1:.*]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_2]]
    // CHECK-DAG: [[PTR1:.+]] = getelementptr inbounds i32, i32*[[ARRAY1]], i64[[INDEX1]]
    // CHECK: store i32[[CALC1]], i32*[[PTR1]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_2]]
    // CHECK-NEXT: br label [[LOOP1_INC:[^,]+]]
    List[i] = i * 2;

    // CHECK: br label [[LOOP1_COND:[^,]+]], !llvm.loop ![[LOOP1_HINTS:[0-9]+]]
  }
}

// Verify assume_safety interleaving is recognized.
void interleave_test(int *List, int Length) {
// CHECK: define {{.*}} @_Z15interleave_test
// CHECK: [[LOAD2_IV:.+]] = load i32, i32* [[IV2:[^,]+]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_8:[0-9]+]]
// CHECK-NEXT: [[LOAD2_LEN:.+]] = load i32, i32* [[LEN2:.+]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_8]]
// CHECK-NEXT: [[CMP2:.+]] = icmp slt i32[[LOAD2_IV]],[[LOAD2_LEN]]
// CHECK-NEXT: br i1[[CMP2]], label %[[LOOP2_BODY:[^,]+]], label %[[LOOP2_END:[^,]+]]
#pragma clang loop interleave(assume_safety) vectorize(disable) unroll(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: [[RHIV2:.+]] = load i32, i32* [[IV2]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_8]]
    // CHECK-DAG: [[CALC2:.+]] = mul nsw i32[[RHIV2]], 2
    // CHECK-DAG: [[SIV2:.+]] = load i32, i32* [[IV2]]{{.*}}!llvm.access.group ![[ACCESS_GROUP_8]]
    // CHECK-DAG: [[INDEX2:.+]] = sext i32[[SIV2]] to i64
    // CHECK-DAG: [[ARRAY2:.+]] = load i32*, i32** [[LIST2:.*]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_8]]
    // CHECK-DAG: [[PTR2:.+]] = getelementptr inbounds i32, i32*[[ARRAY2]], i64[[INDEX2]]
    // CHECK: store i32[[CALC2]], i32*[[PTR2]], {{.*}}!llvm.access.group ![[ACCESS_GROUP_8]]
    // CHECK-NEXT: br label [[LOOP2_INC:[^,]+]]
    List[i] = i * 2;

    // CHECK: br label [[LOOP2_COND:[^,]+]], !llvm.loop ![[LOOP2_HINTS:[0-9]+]]
  }
}

// CHECK: ![[ACCESS_GROUP_2]] = distinct !{}
// CHECK: ![[LOOP1_HINTS]] = distinct !{![[LOOP1_HINTS]], [[MP:![0-9]+]], ![[PARALLEL_ACCESSES_7:[0-9]+]], ![[UNROLL_DISABLE:[0-9]+]], ![[INTERLEAVE_1:[0-9]+]], ![[INTENABLE_1:[0-9]+]]}
// CHECK: ![[PARALLEL_ACCESSES_7]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_2]]}
// CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
// CHECK: ![[INTERLEAVE_1]] = !{!"llvm.loop.interleave.count", i32 1}
// CHECK: ![[INTENABLE_1]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[ACCESS_GROUP_8]] = distinct !{}
// CHECK: ![[LOOP2_HINTS]] = distinct !{![[LOOP2_HINTS]], [[MP]], ![[PARALLEL_ACCESSES_11:[0-9]+]], ![[UNROLL_DISABLE]], ![[WIDTH_1:[0-9]+]], ![[INTENABLE_1]]}
// CHECK: ![[PARALLEL_ACCESSES_11]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_8]]}
// CHECK: ![[WIDTH_1]] = !{!"llvm.loop.vectorize.width", i32 1}
