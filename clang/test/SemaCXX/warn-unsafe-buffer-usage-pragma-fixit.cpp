// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void basic(int * x) {
  int tmp;
  int *p1 = new int[10];  // no fix
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:
  int *p2 = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p2"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
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
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p2"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

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
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p2"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"
  int *p3 = new int[10];
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"std::span<int> p3"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:13}:"{"
  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-3]]:24-[[@LINE-3]]:24}:", 10}"

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
