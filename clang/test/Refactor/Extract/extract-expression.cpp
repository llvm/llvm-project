
struct Rectangle { int width, height; };

int sumArea(Rectangle *rs, int count) {
  int sum = 0;
  for (int i = 0; i < count; ++i) {
    Rectangle r = rs[i];
    sum += r.width * r.height;
  }
  return sum;
}

// RUN: clang-refactor-test perform -action extract -selected=%s:8:12-8:30 %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: "static int extracted(const Rectangle &r) {\nreturn r.width * r.height;\n}\n\n" 4:1 -> 4:1
// CHECK1-NEXT: "extracted(r)" 8:12 -> 8:30
;
void extractFullExpressionIfPartiallySelected(const Rectangle &r) {
  int area = r.width * r.height;
}
// CHECK2: "static int extracted(const Rectangle &r) {\nreturn r.width * r.height;\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK2-NEXT: "extracted(r)" [[@LINE-3]]:14 -> [[@LINE-3]]:32

// RUN: clang-refactor-test perform -action extract -selected=%s:18:20-18:25 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:18:15-18:32 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:18:14-18:30 %s | FileCheck --check-prefix=CHECK2 %s
;
int extractFullMultipleCandidates(const Rectangle &r1) {
  int y = r1.width - r1.width * r1.height;
}
// CHECK3-1: "static int extracted(const Rectangle &r1) {\nreturn - r1.width * r1.height;\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK3-1-NEXT: "+ extracted(r1)" [[@LINE-3]]:20 -> [[@LINE-3]]:42
// CHECK3-2: "static int extracted(const Rectangle &r1) {\nreturn r1.width - r1.width * r1.height;\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK3-2-NEXT: "extracted(r1)" [[@LINE-5]]:11 -> [[@LINE-5]]:42

// RUN: clang-refactor-test perform -action extract -selected=%s:28:20-28:42 %s | FileCheck --check-prefix=CHECK3-1 %s
// RUN: clang-refactor-test perform -action extract -candidate 0 -selected=%s:28:20-28:42 %s | FileCheck --check-prefix=CHECK3-1 %s
// RUN: clang-refactor-test perform -action extract -candidate 1 -selected=%s:28:20-28:42 %s | FileCheck --check-prefix=CHECK3-2 %s

// RUN: not clang-refactor-test perform -action extract -candidate 2 -selected=%s:28:20-28:42 %s 2>&1 | FileCheck --check-prefix=CHECK-CAND-FAIL %s
// CHECK-CAND-FAIL: failed to select the refactoring candidate
;
int extractFullMultipleCandidatesCaptureJustExtractedVariables(
    const Rectangle &r1, const Rectangle &r2) {
  return r1.width - r2.width * r2.height;
}
// CHECK4: "static int extracted(const Rectangle &r2) {\nreturn - r2.width * r2.height;\n}\n\n" [[@LINE-4]]:1 -> [[@LINE-4]]:1
// CHECK4-NEXT: "+ extracted(r2)" [[@LINE-3]]:19 -> [[@LINE-3]]:41

// RUN: clang-refactor-test perform -action extract -candidate 0 -selected=%s:44:19-44:41 %s | FileCheck --check-prefix=CHECK4 %s

// Even when the expression result is unused statement, we still want to extract
// it as an expression.
;
void extractStatementExpression(const Rectangle &r) {
  r.width * r.height;
}
// CHECK5: "static int extracted(const Rectangle &r) {\nreturn r.width * r.height;\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK5-NEXT: "extracted(r)" [[@LINE-3]]:3 -> [[@LINE-3]]:21

// RUN: clang-refactor-test perform -action extract -selected=%s:55:3-55:21 %s | FileCheck --check-prefix=CHECK5 %s
;
void extractFunctionCall() {
  sumArea(0, 1);
}
// CHECK6: "static int extracted() {\nreturn sumArea(0, 1);\n}\n\n" [[@LINE-3]]:1 -> [[@LINE-3]]:1
// CHECK6-NEXT: "extracted()" [[@LINE-3]]:3 -> [[@LINE-3]]:16

// RUN: clang-refactor-test perform -action extract -selected=%s:63:3-63:10 %s | FileCheck --check-prefix=CHECK6 %s
