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

  new S;
}

// CHECK: \"location_context\": \"#0 Call\", \"calling\": \"foo\", \"location\": null, \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"stmt_id\": {{[0-9]+}}, \"kind\": \"construct into local variable\", \"argument_index\": null, \"pretty\": \"T t;\", \"value\": \"&t\"

// CHECK: \"location_context\": \"#0 Call\", \"calling\": \"T::T\", \"location\": \{ \"line\": 16, \"column\": 5, \"file\": \"{{.*}}dump_egraph.cpp\" \}, \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"init_id\": {{[0-9]+}}, \"kind\": \"construct into member variable\", \"argument_index\": null, \"pretty\": \"s\", \"value\": \"&t.s\"

// CHECK: \"cluster\": \"t\", \"pointer\": \"{{0x[0-9a-f]+}}\", \"items\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"Default\", \"offset\": 0, \"value\": \"conj_$2\{int, LC5, no stmt, #1\}\"

// CHECK: \"dynamic_types\": [\l&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\{ \"region\": \"HeapSymRegion\{conj_$1\{struct S *, LC1, S{{[0-9]+}}, #1\}\}\", \"dyn_type\": \"struct S\", \"sub_classable\": false \}\l

