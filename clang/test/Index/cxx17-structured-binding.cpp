// Test is line- and column-sensitive; see below.
int main() {
  int a[2] = {1, 2};
  auto [x, y] = a;
}

// RUN: c-index-test -test-load-source all -std=c++17 %s | FileCheck -check-prefix=CHECK-LOAD %s
// CHECK-LOAD: cxx17-structured-binding.cpp:2:5: FunctionDecl=main:2:5 (Definition) Extent=[2:1 - 5:2]
// CHECK-LOAD: cxx17-structured-binding.cpp:2:12: CompoundStmt= Extent=[2:12 - 5:2]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:3: DeclStmt= Extent=[3:3 - 3:21]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:7: VarDecl=a:3:7 (Definition) Extent=[3:3 - 3:20]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:9: IntegerLiteral= Extent=[3:9 - 3:10]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:14: InitListExpr= Extent=[3:14 - 3:20]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:15: IntegerLiteral= Extent=[3:15 - 3:16]
// CHECK-LOAD: cxx17-structured-binding.cpp:3:18: IntegerLiteral= Extent=[3:18 - 3:19]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:3: DeclStmt= Extent=[4:3 - 4:19]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:8: UnexposedDecl=[x, y]:4:8 (Definition) Extent=[4:3 - 4:18]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:9: UnexposedDecl=x:4:9 (Definition) Extent=[4:9 - 4:10]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:12: UnexposedDecl=y:4:12 (Definition) Extent=[4:12 - 4:13]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: UnexposedExpr= Extent=[4:17 - 4:18]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: DeclRefExpr=a:3:7 Extent=[4:17 - 4:18]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: UnexposedExpr= Extent=[4:17 - 4:9]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: ArraySubscriptExpr= Extent=[4:17 - 4:9]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: UnexposedExpr=a:3:7 Extent=[4:17 - 4:18]
// CHECK-LOAD: cxx17-structured-binding.cpp:4:17: DeclRefExpr=a:3:7 Extent=[4:17 - 4:18]
