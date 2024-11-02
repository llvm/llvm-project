// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: clang-import-test -x objective-c -objc-arc -import %S/Inputs/cleanup-objects.m -dump-ast -expression %s | FileCheck %s

// CHECK: FunctionDecl {{.*}} getObj '
// CHECK: ExprWithCleanups
// CHECK-NEXT: cleanup CompoundLiteralExpr

extern int getObj();
void test(int c, id a) {
  (void)getObj(c, a);
}

