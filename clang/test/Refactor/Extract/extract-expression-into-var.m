// RUN: clang-refactor-test perform -action extract-expression -selected=expr1 -selected=expr2 %s | FileCheck %s

void extractExpressionIntoVar(int x, int y, int z) {
  // expr1-begin: +1:9
  (void)@selector(foo:bar:); // CHECK: "SEL extractedExpr = @selector(foo:bar:);\n" [[@LINE]]:3 -> [[@LINE]]:3
  // expr1-end: -1:28        // CHECK: "extractedExpr" [[@LINE-1]]:9 -> [[@LINE-1]]:28
  // expr2-begin: +1:4       // CHECK: "void (^extractedExpr)(void) = ^ {\n    // do nothing\n  };\n" [[@LINE+1]]:3 -> [[@LINE+1]]:3
  (^ {
    // do nothing
  })();
  // expr2-end: -1:4         // CHECK: "extractedExpr" [[@LINE-3]]:4 -> [[@LINE-1]]:4
}
