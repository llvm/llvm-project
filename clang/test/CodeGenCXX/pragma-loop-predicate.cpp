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

// CHECK:      ![[LOOP0]] = distinct !{![[LOOP0]], !3}
// CHECK-NEXT: !3 = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-NEXT: ![[LOOP1]] = distinct !{![[LOOP1]], !5, !3}
// CHECK-NEXT: !5 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
// CHECK-NEXT: ![[LOOP2]] = distinct !{![[LOOP2]], !7, !3}
// CHECK-NEXT: !7 = !{!"llvm.loop.vectorize.predicate.enable", i1 false}
