// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s
void expr() {
  f();
}

// CHECK: FunctionDecl 0x{{[^ ]*}} <{{[^>]*}}> line:{{.*}}:{{[^ ]*}} used f 'void ()'
// CHECK: -CallExpr 0x{{[^ ]*}} <{{[^>]*}}> 'void' adl
// CHECK: -CXXOperatorCallExpr 0x{{[^ ]*}} <{{[^>]*}}> 'void' '+' adl
