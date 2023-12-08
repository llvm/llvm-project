// Test is line- and column-sensitive; see below.

void foo() {
  if (bool bar = true; bar) {
  }
}

// RUN: c-index-test -test-load-source all -std=c++17 %s | FileCheck -check-prefix=CHECK-LOAD %s
// CHECK-LOAD: cxx17-if-with-initializer.cpp:3:6: FunctionDecl=foo:3:6 (Definition) Extent=[3:1 - 6:2]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:3:12: CompoundStmt= Extent=[3:12 - 6:2]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:3: IfStmt= Extent=[4:3 - 5:4]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:7: DeclStmt= Extent=[4:7 - 4:23]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:12: VarDecl=bar:4:12 (Definition) Extent=[4:7 - 4:22]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:18: CXXBoolLiteralExpr= Extent=[4:18 - 4:22]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:24: UnexposedExpr=bar:4:12 Extent=[4:24 - 4:27]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:24: DeclRefExpr=bar:4:12 Extent=[4:24 - 4:27]
// CHECK-LOAD: cxx17-if-with-initializer.cpp:4:29: CompoundStmt= Extent=[4:29 - 5:4]
