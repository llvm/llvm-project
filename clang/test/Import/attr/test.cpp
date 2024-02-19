// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s
// CHECK: FunctionDecl
// CHECK-SAME: S.cpp:1:1, col:38
// CHECK-NEXT: ConstAttr
// CHECK-SAME: col:32

// CHECK: IndirectFieldDecl
// CHECK-NEXT: Field
// CHECK-NEXT: Field
// CHECK-NEXT: PackedAttr
// CHECK-SAME: col:26

// CHECK: AttributedStmt
// CHECK-NEXT: LoopHintAttr
// CHECK-SAME: line:10:9

extern char f() __attribute__((const));

struct S;

void stmt();

void expr() {
  (void)f();
  struct S s;
}
