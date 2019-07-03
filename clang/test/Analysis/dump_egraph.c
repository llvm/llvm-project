// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:  -analyzer-dump-egraph=%t.dot %s
// RUN: cat %t.dot | FileCheck %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:  -analyzer-dump-egraph=%t.dot \
// RUN:  -trim-egraph %s
// RUN: cat %t.dot | FileCheck %s

// REQUIRES: asserts

int getJ();

int foo() {
  int *x = 0, *y = 0;
  char c = '\x13';

  return *x + *y;
}

// CHECK: \"program_points\": [\l&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"Edge\", \"src_id\": 2, \"dst_id\": 1, \"terminator\": null, \"term_kind\": null, \"tag\": null \}\l&nbsp;&nbsp;],\l&nbsp;&nbsp;\"program_state\": null

// CHECK: \"program_points\": [\l&nbsp;&nbsp;&nbsp;&nbsp;\{ \"kind\": \"BlockEntrance\", \"block_id\": 1


// CHECK: \"pretty\": \"*x\", \"location\": \{ \"line\": 18, \"column\": 10, \"file\": \"{{(.+)}}dump_egraph.c\" \}

// CHECK: \"pretty\": \"'\\\\x13'\"

// CHECK: \"has_report\": true
