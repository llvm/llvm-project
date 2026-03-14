// RUN: %clang_cc1 -ast-dump -fblocks %s | FileCheck -strict-whitespace %s

struct A {};

struct A f1(void) {
  // CHECK:      FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:{{[^:]*}}:1> line:[[@LINE-1]]:10 f1 'struct A (void)'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:19, line:{{[^:]*}}:1>
  struct A a;
  // CHECK-NEXT: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13>
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:12> col:12 used a 'struct A' nrvo
  return a;
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10>
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:10> 'struct A' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'struct A' lvalue Var 0x{{[^ ]*}} 'a' 'struct A'
}

void f2(void) {
  (void)^{
    // CHECK:      BlockDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:9, line:{{[^:]*}}:3> line:[[@LINE-1]]:9
    // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:10, line:{{[^:]*}}:3>
    struct A a;
    // CHECK-NEXT: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:15>
    // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:5, col:14> col:14 used a 'struct A' nrvo
    return a;
    // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:12>
    // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:12> 'struct A' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'struct A' lvalue Var 0x{{[^ ]*}} 'a' 'struct A'
  }();
}
