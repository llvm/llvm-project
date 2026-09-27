// Test is line- and column-sensitive; see below.

void foo() {
  switch (int bar = true; bar) {
  }
}

// RUN: c-index-test -test-load-source all -std=c++17 %s | FileCheck -check-prefix=CHECK %s
// CHECK: cxx17-switch-with-initializer.cpp:3:6: FunctionDecl=foo:3:6 (Definition) Extent=[3:1 - 6:2]
// CHECK: cxx17-switch-with-initializer.cpp:3:12: CompoundStmt= Extent=[3:12 - 6:2]
// CHECK: cxx17-switch-with-initializer.cpp:4:3: SwitchStmt= Extent=[4:3 - 5:4]
// CHECK: cxx17-switch-with-initializer.cpp:4:11: DeclStmt= Extent=[4:11 - 4:26]
// CHECK: cxx17-switch-with-initializer.cpp:4:15: VarDecl=bar:4:15 (Definition) Extent=[4:11 - 4:25]
// CHECK: cxx17-switch-with-initializer.cpp:4:21: UnexposedExpr= Extent=[4:21 - 4:25]
// CHECK: cxx17-switch-with-initializer.cpp:4:21: CXXBoolLiteralExpr= Extent=[4:21 - 4:25]
// CHECK: cxx17-switch-with-initializer.cpp:4:27: UnexposedExpr=bar:4:15 Extent=[4:27 - 4:30]
// CHECK: cxx17-switch-with-initializer.cpp:4:27: DeclRefExpr=bar:4:15 Extent=[4:27 - 4:30]
// CHECK: cxx17-switch-with-initializer.cpp:4:32: CompoundStmt= Extent=[4:32 - 5:4]
