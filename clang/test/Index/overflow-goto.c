// RUN: c-index-test -test-load-source all -Xclang -fexperimental-overflow-behavior-types %s | FileCheck %s

int test(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}

// CHECK: overflow-goto.c:4:10: UnexposedExpr= Extent=[4:10 - 4:43]
// CHECK: overflow-goto.c:4:34: BinaryOperator=+ Extent=[4:34 - 4:39]
// CHECK: overflow-goto.c:4:41: LabelRef=l:5:1 Extent=[4:41 - 4:42]
// CHECK: overflow-goto.c:5:1: LabelStmt=l Extent=[5:1 - 6:11]
