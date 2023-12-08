// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

void loop1(int *List, int Length) {
// CHECK-LABEL: @{{.*}}loop1{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP1:.*]]

  #pragma clang loop vectorize(enable) vectorize_width(1)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Here, vectorize.enable should be set, obviously, but also check that
// metadata isn't added twice.
void loop2(int *List, int Length) {
// CHECK-LABEL: @{{.*}}loop2{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP2:.*]]

  #pragma clang loop vectorize(enable) vectorize_width(2)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Test that we do *not* imply vectorize.enable.
void loop3(int *List, int Length) {
// CHECK-LABEL: @{{.*}}loop3{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP3:.*]]

  #pragma clang loop vectorize_width(1)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// Test that we *do* imply vectorize.enable.
void loop4(int *List, int Length) {
// CHECK-LABEL: @{{.*}}loop4{{.*}}(
// CHECK: br label {{.*}}, !llvm.loop ![[LOOP4:.*]]

  #pragma clang loop vectorize_width(2)
  for (int i = 0; i < Length; i++)
    List[i] = i * 2;
}

// CHECK: ![[LOOP1]] = distinct !{![[LOOP1]], [[MP:![0-9]+]], ![[VEC_WIDTH_1:.*]], ![[FIXED_WIDTH:.*]], ![[VEC_ENABLE:.*]]}
// CHECK: ![[VEC_WIDTH_1]] = !{!"llvm.loop.vectorize.width", i32 1}
// CHECK: ![[FIXED_WIDTH]] = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
// CHECK: ![[VEC_ENABLE]] = !{!"llvm.loop.vectorize.enable", i1 true}

// CHECK: ![[LOOP2]] = distinct !{![[LOOP2]], [[MP]], ![[VEC_WIDTH_2:.*]], ![[FIXED_WIDTH:.*]], ![[VEC_ENABLE]]}
// CHECK: ![[VEC_WIDTH_2]] = !{!"llvm.loop.vectorize.width", i32 2}

// CHECK: ![[LOOP3]] = distinct !{![[LOOP3]], [[MP]], ![[VEC_WIDTH_1]], ![[FIXED_WIDTH]]}

// CHECK: ![[LOOP4]] = distinct !{![[LOOP4]], [[MP]], ![[VEC_WIDTH_2]], ![[FIXED_WIDTH]], ![[VEC_ENABLE]]}
