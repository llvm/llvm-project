// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s 2>&1 | FileCheck %s
// expected-no-diagnostics

void clang_analyzer_printState();

struct Member {
  int large[10];
};
Member getMember();

struct Class {
  Member m;
  int first;
  int second;
  int third;
};


void test_output(int n) {
  Class objsecond;
  objsecond.m.large[n] = 20;

  Class objfirst;

  objfirst.m = getMember();
  objfirst.second = 2;
  objfirst.third = 3;
  objfirst.first = 1;

  clang_analyzer_printState();
  // Default binding is before any direct bindings.
  // Direct bindings are increasing by offset.
  // Global memory space clusters come before any other clusters.
  // Otherwise, Clusters are in alphabetical order.

  // CHECK:       "store": { "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:    { "cluster": "GlobalInternalSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "GlobalSystemSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "objfirst", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:      { "kind": "Direct", "offset": 320, "value": "1 S32b" },
  // CHECK-NEXT:      { "kind": "Direct", "offset": 352, "value": "2 S32b" },
  // CHECK-NEXT:      { "kind": "Direct", "offset": 384, "value": "3 S32b" }
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "objsecond", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "Unknown" },
  // CHECK-NEXT:      { "kind": "Direct", "offset": null, "value": "20 S32b" }
  // CHECK-NEXT:    ]}
  // CHECK-NEXT:  ]},

  (void)objfirst;
  (void)objsecond;
}
