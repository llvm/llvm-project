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

// CHECK: \"program_points\": [\l
// CHECK-SAME: \{ \"kind\": \"Edge\", \"src_id\": 2, \"dst_id\": 1,
// CHECK-SAME:    \"terminator\": null, \"term_kind\": null, \"tag\": null,
// CHECK-SAME:    \"node_id\": 1, \"is_sink\": 0, \"has_report\": 0
// CHECK-SAME: \},
// CHECK-SAME: \{ \"kind\": \"BlockEntrance\", \"block_id\": 1, \"tag\": null,
// CHECK-SAME:    \"node_id\": 2, \"is_sink\": 0, \"has_report\": 0
// CHECK-SAME: \},
// CHECK-SAME: \{ \"kind\": \"Statement\", \"stmt_kind\": \"IntegerLiteral\",
// CHECK-SAME:    \"stmt_id\": {{[0-9]*}}, \"pointer\": \"0x{{[0-9a-f]*}}\",
// CHECK-SAME:    \"pretty\": \"0\", \"location\": \{
// CHECK-SAME:        \"line\": 15, \"column\": 12, \"file\":
// CHECK-SAME:    \}, \"stmt_point_kind\": \"PreStmtPurgeDeadSymbols\",
// CHECK-SAME:    \"tag\": \"ExprEngine : Clean Node\", \"node_id\": 3,
// CHECK-SAME:    \"is_sink\": 0, \"has_report\": 0
// CHECK-SAME: \},
// CHECK-SAME: \{ \"kind\": \"Statement\", \"stmt_kind\": \"IntegerLiteral\",
// CHECK-SAME:    \"stmt_id\": {{[0-9]*}}, \"pointer\": \"0x{{[0-9a-f]*}}\",
// CHECK-SAME:    \"pretty\": \"0\", \"location\": \{
// CHECK-SAME:        \"line\": 15, \"column\": 12, \"file\":
// CHECK-SAME:    \}, \"stmt_point_kind\": \"PostStmt\", \"tag\": null,
// CHECK-SAME:    \"node_id\": 4, \"is_sink\": 0, \"has_report\": 0
// CHECK-SAME: \}
// CHECK-SAME: ]

// CHECK: \"pretty\": \"*x\", \"location\": \{ \"line\": 18, \"column\": 10, \"file\": \"{{(.+)}}dump_egraph.c\" \}

// CHECK: \"pretty\": \"'\\\\x13'\"

// CHECK: \"has_report\": 1

// CHECK-NOT: \"program_state\": null
