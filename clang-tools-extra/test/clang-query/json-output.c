// RUN: clang-query -c "set output json" -c "match functionDecl()" %s -- | FileCheck %s

// CHECK: {"matcher":"functionDecl()","match_count":1,"matches":[{"bindings":{"root":{"kind":"FunctionDecl"
// CHECK-SAME: "range":{"file":"{{.*}}json-output.c","begin":{"line":
// CHECK-SAME: "detail":{
// CHECK-SAME: "kind":"FunctionDecl"
// CHECK-SAME: "name":"foo"
// CHECK-SAME: "qualType":"void (void)"
void foo(void) {}
