// RUN: %clang_cc1 -ast-dump -fblocks %s | FileCheck -strict-whitespace %s

struct ExplicitBase {
  explicit ExplicitBase(const char *) { }
  ExplicitBase(const ExplicitBase &) {}
  ExplicitBase(ExplicitBase &&) {}
  ExplicitBase &operator=(const ExplicitBase &) { return *this; }
  ExplicitBase &operator=(ExplicitBase &&) { return *this; }
  ~ExplicitBase() { }
};

struct Derived1 : ExplicitBase {};

Derived1 makeDerived1() {
  // CHECK:      FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:{{[^:]*}}:1> line:[[@LINE-1]]:10 makeDerived1 'Derived1 ()'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:{{[^ ^,]+}}, line:{{[^:]*}}:1
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <line:[[@LINE+6]]:3, col:{{[0-9]+}}>
  // CHECK-DAG:  MaterializeTemporaryExpr 0x{{[^ ]*}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'ExplicitBase' xvalue
  // CHECK-NEXT: CXXBindTemporaryExpr 0x[[TEMP:[^ ]*]] <col:{{[0-9]+}}, col:{{[0-9]+}}> 'ExplicitBase' (CXXTemporary 0x[[TEMP]])
  // CHECK-NEXT: CXXTemporaryObjectExpr 0x{{[^ ]*}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'ExplicitBase' 'void (const char *)' list
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:{{[0-9]+}}> 'const char *' <ArrayToPointerDecay>
  // CHECK-NEXT: StringLiteral 0x{{[^ ]*}} <col:{{[0-9]+}}> 'const char[10]' lvalue "Move Ctor"
  return Derived1{ExplicitBase{"Move Ctor"}};
}

struct ImplicitBase {
  ImplicitBase(const char *) { }
  ImplicitBase(const ImplicitBase &) {}
  ImplicitBase(ImplicitBase &&) {}
  ImplicitBase &operator=(const ImplicitBase &) { return *this; }
  ImplicitBase &operator=(ImplicitBase &&) { return *this; }
  ~ImplicitBase() { }
};

struct Derived2 : ImplicitBase {};

Derived2 makeDerived2() {
  // CHECK:      FunctionDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:{{[^:]*}}:1> line:[[@LINE-1]]:10 makeDerived2 'Derived2 ()'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:{{[^ ^,]+}}, line:{{[^:]*}}:1
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <line:[[@LINE+2]]:3, col:{{[0-9]+}}>
  // CHECK-NOT:  MaterializeTemporaryExpr 0x{{[^ ]*}} <col:{{[0-9]+}}, col:{{[0-9]+}}> 'ImplicitBase' xvalue
  return Derived2{{"No Ctor"}};
}
