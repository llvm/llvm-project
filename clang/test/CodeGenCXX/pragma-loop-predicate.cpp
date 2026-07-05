// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

void test0(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test0{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP0:.*]]

  #pragma clang loop vectorize(enable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

void test1(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test1{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP1:.*]]

  #pragma clang loop vectorize(enable) vectorize_predicate(enable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

void test2(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test2{{.*}}(
// CHECK:       br label {{.*}}, !llvm.loop ![[LOOP2:.*]]

  #pragma clang loop vectorize(enable) vectorize_predicate(disable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// vectorize_predicate(enable) implies vectorize(enable)
void test3(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test3{{.*}}(
// CHECK:       br label {{.*}}, !llvm.loop ![[LOOP3:.*]]

  #pragma clang loop vectorize_predicate(enable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Check that disabling vectorization means a vectorization width of 1, and
// also that vectorization_predicate isn't enabled.
void test4(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test4{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP4:.*]]

  #pragma clang loop vectorize(disable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Check that vectorize and vectorize_predicate are disabled.
void test5(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test5{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP5:.*]]

  #pragma clang loop vectorize(disable) vectorize_predicate(enable)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Check that vectorize_predicate is ignored when vectorization width is 1
void test6(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test6{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP6:.*]]

#pragma clang loop vectorize_predicate(disable) vectorize_width(1)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}


// Check that vectorize_width(!=1) does not affect vectorize_predicate.
void test7(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test7{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP7:.*]]

#pragma clang loop vectorize_predicate(disable) vectorize_width(4)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}


// Check that vectorize_predicate is ignored when vectorization width is 1
void test8(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test8{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP8:.*]]

#pragma clang loop vectorize_predicate(enable) vectorize_width(1)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}


// Check that vectorize_width(!=1) does not affect vectorize_predicate.
void test9(int *List, int Length) {
// CHECK-LABEL: @{{.*}}test9{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP9:.*]]

#pragma clang loop vectorize_predicate(enable) vectorize_width(4)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// CHECK:      ![[LOOP0]] = distinct !{![[LOOP0]], [[MP:![0-9]+]], [[GEN3:![0-9]+]]}
// CHECK:      [[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK-NEXT: [[GEN3]] = !{!"llvm.loop.vectorize.enable", i1 true}

// CHECK-NEXT: ![[LOOP1]] = distinct !{![[LOOP1]], [[MP]], [[GEN6:![0-9]+]], [[GEN3]]}
// CHECK-NEXT: [[GEN6]] = !{!"llvm.loop.vectorize.predicate.enable", i1 true}

// CHECK-NEXT: ![[LOOP2]] = distinct !{![[LOOP2]], [[MP]], [[GEN8:![0-9]+]], [[GEN3]]}
// CHECK-NEXT: [[GEN8]] = !{!"llvm.loop.vectorize.predicate.enable", i1 false}

// CHECK-NEXT: ![[LOOP3]] = distinct !{![[LOOP3]], [[MP]], [[GEN6]], [[GEN3]]}

// CHECK-NEXT: ![[LOOP4]] = distinct !{![[LOOP4]], [[MP]], [[GEN10:![0-9]+]]}
// CHECK-NEXT: [[GEN10]] = !{!"llvm.loop.vectorize.width", i32 1}

// CHECK-NEXT: ![[LOOP5]] = distinct !{![[LOOP5]], [[MP]], [[GEN6]], [[GEN10]]}

// CHECK-NEXT: ![[LOOP6]] = distinct !{![[LOOP6]], [[MP]], [[GEN8]], [[GEN10]], [[GEN11:![0-9]+]]}
// CHECK-NEXT: [[GEN11]] = !{!"llvm.loop.vectorize.scalable.enable", i1 false}

// CHECK-NEXT: ![[LOOP7]] = distinct !{![[LOOP7]], [[MP]], [[GEN8]], [[GEN12:![0-9]+]], [[GEN11]], [[GEN3]]}
// CHECK-NEXT: [[GEN12]] = !{!"llvm.loop.vectorize.width", i32 4}

// CHECK-NEXT: ![[LOOP8]] = distinct !{![[LOOP8]], [[MP]], [[GEN6]], [[GEN10]], [[GEN11]]}

// CHECK-NEXT: ![[LOOP9]] = distinct !{![[LOOP9]], [[MP]], [[GEN6]], [[GEN12]], [[GEN11]], [[GEN3]]}
