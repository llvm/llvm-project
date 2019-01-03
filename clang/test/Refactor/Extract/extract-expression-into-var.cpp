int call(int x);

void extractExpressionIntoVar(int x, int y, int z) {
  {
    // expr1-begin: +1:13
    int p = x + y - z;   // CHECK: "int extractedExpr = x + y;\n" [[@LINE]]:5 -> [[@LINE]]:5 [Symbol extracted-decl 0 1:5 -> 1:18]
    // expr1-end: -1:19  // CHECK: "extractedExpr" [[@LINE-1]]:13 -> [[@LINE-1]]:18 [Symbol extracted-decl-ref 0 1:1 -> 1:14]
    // expr2-begin: +1:6 // CHECK: "std::function<void (int)> extractedExpr = [&](int x) {\n       p = x;\n    };\n" [[@LINE+1]]:5 -> [[@LINE+1]]:5 [Symbol extracted-decl 0 1:27 -> 1:40]
    ([&](int x) {
       p = x;
    })(0);
    // expr2-end: -1:6   // CHECK: "extractedExpr" [[@LINE-3]]:6 -> [[@LINE-1]]:6
  }
  #define MACROARG(x) (x)
  // expr3-begin: +1:20
  int p = MACROARG(y + z) - x; // CHECK: "int extractedExpr = y + z;\n" [[@LINE]]:3 -> [[@LINE]]:3
  // expr3-end: -1:25   // CHECK: "extractedExpr" [[@LINE-1]]:20 -> [[@LINE-1]]:25
  #define MACROSTMT(x, y) int var = (x) + (y);
  // expr4-begin: +1:16
  MACROSTMT(0, call(p * z))   // CHECK: "int extractedExpr = call(p * z);\n" [[@LINE]]:3 -> [[@LINE]]:3
  // expr4-end: -1:27   // CHECK: "extractedExpr" [[@LINE-1]]:16 -> [[@LINE-1]]:27
}

// RUN: clang-refactor-test perform -action extract-expression -selected=expr1 -selected=expr2 -selected=expr3 -selected=expr4 -emit-associated %s -std=c++11 | FileCheck %s

// RUN: clang-refactor-test list-actions -at=%s:6:13 -selected=%s:6:13-6:18 %s -std=c++11 | FileCheck --check-prefix=CHECK-ACTION %s
// CHECK-ACTION: Extract Expression

void dontExtractStatement(int x, int y) {
  // stmt1-begin: +1:3
  x = y;
  // stmt1-end: -1:8
  // stmt2-begin: +1:3
  return;
  // stmt2-end: +0:1
}

// RUN: not clang-refactor-test perform -action extract-expression -selected=stmt1 -selected=stmt2 %s -std=c++11 2>&1 | FileCheck --check-prefix=CHECK-FAIL %s
// CHECK-FAIL: Failed to initiate the refactoring action!

void dontExtractVoidCall() {
  // void-call-begin: +1:3
  dontExtractVoidCall();
  // void-call-end: -1:24
}

// RUN: not clang-refactor-test perform -action extract-expression -selected=void-call %s -std=c++11 2>&1 | FileCheck --check-prefix=CHECK-FAIL %s
