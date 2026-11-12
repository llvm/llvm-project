// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void basic(int * x) {
  int tmp;
  int *p1 = new int[10];  // no fix
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p2 = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];
#pragma clang unsafe_buffer_usage end
  tmp = p2[5];
}

void withDiagnosticWarning() {
  int tmp;
  int *p1 = new int[10]; // no fix
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p2 = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

  // diagnostics in opt-out region
#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic warning "-Weverything"
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic pop
#pragma clang unsafe_buffer_usage end

  // opt-out region under diagnostic warning
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
#pragma clang diagnostic pop

  tmp = p2[5];
}


void withDiagnosticIgnore() {
  int tmp;
  int *p1 = new int[10];
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p2 = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  int *p3 = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int> "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic ignored "-Weverything"
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang diagnostic pop
#pragma clang unsafe_buffer_usage end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
#pragma clang diagnostic pop

  tmp = p2[5];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#pragma clang unsafe_buffer_usage begin
  tmp = p1[5];  // not to warn
  tmp = p2[5];  // not to warn
#pragma clang unsafe_buffer_usage end
  tmp = p3[5];  // expected-note{{used in buffer access here}}
#pragma clang diagnostic pop
}

void noteGoesWithVarDeclWarning() {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
  int *p = new int[10]; // not to warn
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
#pragma clang diagnostic pop

  p[5]; // not to note since the associated warning is suppressed
}


// Test suppressing interacts with variable grouping:

// The implication edges are: `a` -> `b` -> `c`.
// If the unsafe operation on `a` is supressed, none of the variables
// will be fixed.
void suppressedVarInGroup() {
  int * a;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * b;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * c;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

#pragma clang unsafe_buffer_usage begin
  a[5] = 5;
#pragma clang unsafe_buffer_usage end
  a = b;
  b = c;
}

// To show that if `a[5]` is not suppressed in the
// `suppressedVarInGroup` function above, all variables will be fixed.
void suppressedVarInGroup_control() {
  int * a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * b;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * c;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"

  a[5] = 5;
  a = b;
  b = c;
}

// The implication edges are: `a` -> `b` -> `c`.
// The unsafe operation on `b` is supressed, while the unsafe
// operation on `a` is not suppressed. Variable `b` will still be
// fixed when fixing `a`.
void suppressedVarInGroup2() {
  int * a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * b;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * c;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"

  a[5] = 5;
#pragma clang unsafe_buffer_usage begin
  b[5] = 5;
#pragma clang unsafe_buffer_usage end
  a = b;
  b = c;
}

// The implication edges are: `a` -> `b` -> `c`.
// The unsafe operation on `b` is supressed, while the unsafe
// operation on `c` is not suppressed. Only variable `c` will be fixed
// then.
void suppressedVarInGroup3() {
  int * a;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * b;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * c;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:

  c[5] = 5;
#pragma clang unsafe_buffer_usage begin
  b[5] = 5;
#pragma clang unsafe_buffer_usage end
  a = b;
  b = c;
// FIXME: we do not fix `a = b` and `b = c` because the `.data()`  fix-it is not generally correct.
}

// The implication edges are: `a` -> `b` -> `c` -> `a`.
// The unsafe operation on `b` is supressed, while the unsafe
// operation on `c` is not suppressed. Since the implication graph
// forms a cycle, all variables will be fixed.
void suppressedVarInGroup4() {
  int * a;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * b;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * c;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"

  c[5] = 5;
#pragma clang unsafe_buffer_usage begin
  b[5] = 5;
#pragma clang unsafe_buffer_usage end
  a = b;
  b = c;
  c = a;
}

// There are two groups: `a` -> `b` -> `c` and `d` -> `e` -> `f`.
// Suppressing unsafe operations on variables in one group does not
// affect other groups.
void suppressedVarInGroup5() {
  int * a;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * b;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * c;
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int * d;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * e;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"
  int * f;
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:8}:"std::span<int>"

#pragma clang unsafe_buffer_usage begin
  a[5] = 5;
#pragma clang unsafe_buffer_usage end
  a = b;
  b = c;

  d[5] = 5;
  d = e;
  e = f;
}
