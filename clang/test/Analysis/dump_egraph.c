// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-dump-egraph=%t.dot -trim-egraph %s
// REQUIRES: asserts

int getJ();

int foo() {
  int *x = 0, *y = 0;
  return *x + *y;
}

// CHECK: digraph "Exploded Graph" {

// CHECK: \"program_points\": [\l&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"Edge\", \"src_id\": 2, \"dst_id\": 1, \"terminator\": null, \"term_kind\": null, \"tag\": null \}\l&nbsp;&nbsp;],\l&nbsp;&nbsp;\"program_state\": null

// CHECK: \"program_points\": [\l&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"BlockEntrance\", \"block_id\": 1

// CHECK: \"has_report\": true

