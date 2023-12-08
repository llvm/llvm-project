// Test is line- and column-sensitive. Run lines are below.

struct Foo {
  int x = 10;
};

void foo() {
  int p = 11;
}

#define FUNC_MAC(x) x

void goo() {
  int p = FUNC_MAC(1);
  int a = __LINE__;
}

unsigned long long foo_int = 1ull << 60;

unsigned long long HUGE = 1ull << 63;

long long HUGE_NEG = -(1ll << 35);

template <typename d> class e {
  using f = d;
  static const auto g = alignof(f);
};

constexpr static int calc_val() { return 1 + 2; }
const auto the_value = calc_val() + sizeof(char);

void vlaTest() {
  int msize = 4;
  float arr[msize];
  [&arr] {};
}

// RUN: c-index-test -evaluate-cursor-at=%s:4:7 \
// RUN:    -evaluate-cursor-at=%s:8:7 \
// RUN:    -evaluate-cursor-at=%s:8:11 -std=c++11 %s | FileCheck %s
// CHECK: Value: 10
// CHECK: Value: 11
// CHECK: Value: 11

// RUN: c-index-test -get-macro-info-cursor-at=%s:11:9 \
// RUN:    -get-macro-info-cursor-at=%s:14:11 \
// RUN:    -get-macro-info-cursor-at=%s:15:11 -std=c++11 %s | FileCheck -check-prefix=CHECK-MACRO %s
// CHECK-MACRO: [function macro]
// CHECK-MACRO: [function macro]
// CHECK-MACRO: [builtin macro]

// RUN: c-index-test -evaluate-cursor-at=%s:18:20 \
// RUN:    -evaluate-cursor-at=%s:20:20 \
// RUN:    -evaluate-cursor-at=%s:22:11 \
// RUN:    -std=c++11 %s | FileCheck -check-prefix=CHECK-LONG %s
// CHECK-LONG: unsigned, Value: 1152921504606846976
// CHECK-LONG: unsigned, Value: 9223372036854775808
// CHECK-LONG: Value: -34359738368

// RUN: c-index-test -evaluate-cursor-at=%s:18:20 \
// RUN:    -evaluate-cursor-at=%s:20:20 \
// RUN:    -evaluate-cursor-at=%s:26:21 \
// RUN:    -std=c++11 %s | FileCheck -check-prefix=CHECK-DOES-NOT-CRASH %s
// CHECK-DOES-NOT-CRASH: Not Evaluatable

// RUN: c-index-test -evaluate-cursor-at=%s:30:1 \
// RUN:    -evaluate-cursor-at=%s:30:32 \
// RUN:    -evaluate-cursor-at=%s:30:35 \
// RUN:    -evaluate-cursor-at=%s:30:37 -std=c++11 %s | FileCheck %s -check-prefix=CHECK-EXPR
// CHECK-EXPR: unsigned, Value: 4
// CHECK-EXPR: Value: 3
// CHECK-EXPR: unsigned, Value: 4
// CHECK-EXPR: unsigned, Value: 1

// RUN: c-index-test -evaluate-cursor-at=%s:35:5 \
// RUN:    -std=c++11 %s | FileCheck -check-prefix=VLA %s
// VLA: Not Evaluatable
