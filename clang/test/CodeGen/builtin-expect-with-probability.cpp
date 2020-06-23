// RUN: %clang_cc1 -emit-llvm -o - %s -O1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -fexperimental-new-pass-manager %s -O1 | FileCheck %s
extern int global;

struct S {
  static constexpr int prob = 1;
};

template<typename T>
int expect_taken(int x) {
// CHECK: !{{[0-9]+}} = !{!"branch_weights", i32 2147483647, i32 1}

	if (__builtin_expect_with_probability (x == 100, 1, T::prob)) {
		return 0;
	}
	return x;
}

int f() {
  return expect_taken<S>(global);
}

int expect_taken2(int x) {
  // CHECK: !{{[0-9]+}} = !{!"branch_weights", i32 1932735283, i32 214748366}

  if (__builtin_expect_with_probability(x == 100, 1, 0.9)) {
    return 0;
  }
  return x;
}

int expect_taken3(int x) {
  // CHECK: !{{[0-9]+}} = !{!"branch_weights", i32 107374184, i32 107374184, i32 1717986918, i32 107374184, i32 107374184}
  switch (__builtin_expect_with_probability(x, 1, 0.8)) {
  case 0:
    x = x + 0;
  case 1:
    x = x + 1;
  case 2:
    x = x + 2;
  case 5:
    x = x + 5;
  default:
    x = x + 6;
  }
  return x;
}
