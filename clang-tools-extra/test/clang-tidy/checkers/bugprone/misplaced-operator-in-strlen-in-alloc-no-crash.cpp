// RUN: %check_clang_tidy -expect-clang-tidy-error %s bugprone-misplaced-operator-in-strlen-in-alloc %t

void *f() { return new int[](); }
// CHECK-MESSAGES: :[[@LINE-1]]:24: error: cannot determine allocated array size from initializer [clang-diagnostic-error]

template <int... Is> void g() {
  new int[]{Is...};
}
