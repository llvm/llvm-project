// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s
// REQUIRES: asserts

struct S {
  ~S();
};

struct T {
  S s;
  T() : s() {}
};

void foo() {
  // Test that dumping symbols conjured on null statements doesn't crash.
  T t;
}

// CHECK: \"location_context\": \"#0 Call\", \"calling\": \"foo\", \"call_line\": null, \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"lctx_id\": 1, \"stmt_id\": {{[0-9]+}}, \"kind\": \"construct into local variable\", \"argument_index\": null, \"pretty\": \"T t;\", \"value\": \"&t\"

// CHECK: \"location_context\": \"#0 Call\", \"calling\": \"T::T\", \"call_line\": \"16\", \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"lctx_id\": 2, \"init_id\": {{[0-9]+}}, \"kind\": \"construct into member variable\", \"argument_index\": null, \"pretty\": \"s\", \"value\": \"&t-\>s\"

// CHECK: \"cluster\": \"t\", \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"Default\", \"offset\": 0, \"value\": \"conj_$3\{int, LC3, no stmt, #1\}\"

