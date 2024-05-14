// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Check that passing -fno-unroll-loops does not impact the decision made using pragmas.
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - -O1 -disable-llvm-optzns -fno-unroll-loops %s | FileCheck %s

// Verify while loop is recognized after unroll pragma.
void while_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z10while_test
  int i = 0;

#pragma GCC unroll
  while (i < Length) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
    List[i] = i * 2;
    i++;
  }
}

// Verify do loop is recognized after multi-option pragma clang loop directive.
void do_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z7do_test
  int i = 0;

#pragma GCC nounroll
  do {
    // CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
    List[i] = i * 2;
    i++;
  } while (i < Length);
}

// Verify for loop is recognized after unroll pragma.
void for_test(int *List, int Length) {
// CHECK: define {{.*}} @_Z8for_test
#pragma GCC unroll 8
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
    List[i] = i * 2;
  }
}

// Verify c++11 for range loop is recognized after unroll pragma.
void for_range_test() {
  // CHECK: define {{.*}} @_Z14for_range_test
  double List[100];

#pragma GCC unroll(4)
  for (int i : List) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
    List[i] = i;
  }
}

#define UNROLLCOUNT 8

// Verify defines are correctly resolved in unroll pragmas.
void for_define_test(int *List, int Length, int Value) {
// CHECK: define {{.*}} @_Z15for_define_test
#pragma GCC unroll(UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
    List[i] = i * Value;
  }
}

// Verify metadata is generated when template is used.
template <typename A>
void for_template_test(A *List, int Length, A Value) {
// CHECK: define {{.*}} @_Z13template_test
#pragma GCC unroll 8
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_6:.*]]
    List[i] = i * Value;
  }
}

// Verify define is resolved correctly when template is used.
template <typename A>
void for_template_define_test(A *List, int Length, A Value) {
// CHECK: define {{.*}} @_Z24for_template_define_test

#pragma GCC unroll(UNROLLCOUNT)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_7:.*]]
    List[i] = i * Value;
  }
}

#undef UNROLLCOUNT

// Use templates defined above. Test verifies metadata is generated correctly.
void template_test(double *List, int Length) {
  double Value = 10;

  for_template_test<double>(List, Length, Value);
  for_template_define_test<double>(List, Length, Value);
}

void for_unroll_zero_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z20for_unroll_zero_testPii
  #pragma GCC unroll 0
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_14:.*]]
    List[i] = i * 2;
  }
}

void while_unroll_zero_test(int *List, int Length) {
  // CHECK: define {{.*}} @_Z22while_unroll_zero_testPii
  int i = 0;
#pragma GCC unroll(0)
  while (i < Length) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_15:.*]]
    List[i] = i * 2;
    i++;
  }
}

using size_t = unsigned long long;

template <bool Flag>
int value_dependent(int n) {
  // CHECK: define {{.*}} @_Z15value_dependentILb1EEii
  constexpr int N = 100;
  auto init = [=]() { return Flag ? n : 0UL; };
  auto cond = [=](size_t ix) { return Flag ? ix != 0 : ix < 10; };
  auto iter = [=](size_t ix) {
    return Flag ? ix & ~(1ULL << __builtin_clzll(ix)) : ix + 1;
  };
#pragma GCC unroll Flag ? 1 : N
  for (size_t ix = init(); cond(ix); ix = iter(ix)) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_16:.*]]
    n *= n;
  }
#pragma GCC unroll Flag ? 0 : N
  for (size_t ix = init(); cond(ix); ix = iter(ix)) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_17:.*]]
    n *= n;
  }
  return n;
}

void test_value_dependent(int n) {
  value_dependent<true>(n);
}

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], [[MP:![0-9]+]], ![[UNROLL_ENABLE:.*]]}
// CHECK: ![[UNROLL_ENABLE]] = !{!"llvm.loop.unroll.enable"}
// CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2:.*]], ![[UNROLL_DISABLE:.*]]}
// CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
// CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], [[MP]], ![[UNROLL_8:.*]]}
// CHECK: ![[UNROLL_8]] = !{!"llvm.loop.unroll.count", i32 8}
// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[UNROLL_4:.*]]}
// CHECK: ![[UNROLL_4]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: ![[LOOP_5]] = distinct !{![[LOOP_5]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_6]] = distinct !{![[LOOP_6]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_7]] = distinct !{![[LOOP_7]], ![[UNROLL_8:.*]]}
// CHECK: ![[LOOP_14]] = distinct !{![[LOOP_14]], [[MP]], ![[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOOP_15]] = distinct !{![[LOOP_15]], [[MP]], ![[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOOP_16]] = distinct !{![[LOOP_16]], [[MP]], ![[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOOP_17]] = distinct !{![[LOOP_17]], [[MP]], ![[UNROLL_DISABLE:.*]]}
