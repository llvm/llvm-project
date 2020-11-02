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

// CHECK-NEXT: ![[LOOP5]] = distinct !{![[LOOP5]], [[MP]], [[GEN10]]}
