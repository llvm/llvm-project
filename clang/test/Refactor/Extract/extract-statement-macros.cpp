#define INT int
#define MACRO INT y = x * x

void extractStatementsTrimComments(int x) {
  INT y = 0;

  // comment
  MACRO;

  int z = 0;
}
// CHECK1: Initiated the 'extract' action at [[@LINE-4]]:3 -> [[@LINE-2]]:13
// CHECK2: Initiated the 'extract' action at [[@LINE-8]]:3 -> [[@LINE-5]]:9

// RUN: clang-refactor-test initiate -action extract -selected=%s:6:1-10:12 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract -selected=%s:5:3-9:1 %s | FileCheck --check-prefix=CHECK2 %s

#define BLOCK __attribute__((__blocks__(byref)))
void macroAtDeclStmt() {
  // attr-begin: +1:38
  BLOCK const char *Message = "HELLO";
  int X = 123;
  // attr-end: -1:13
}
// CHECK3: Initiated the 'extract' action at [[@LINE-4]]:3 -> [[@LINE-3]]:15
// RUN: clang-refactor-test initiate -action extract -selected=attr %s -fblocks | FileCheck --check-prefix=CHECK3 %s

#define MUT(x) (--(x))

void macroExtractionEndsInMacroArgument(int x, int y) { // CHECK4: "static void extracted(int &x, int &y) {\ny = MUT(x);\n}\n\n" [[@LINE]]:1
// CHECK5: "static int extracted(int &x) {\nreturn MUT(x);\n}\n\n" [[@LINE-1]]:1

  // macro-arg-expr-begin: +3:7
  // macro-arg-end1-begin: +2:1
  // macro-arg-end2-begin: +1:1
  y = MUT(x); // comment
  // macro-arg-end1-end: -1:25
  // macro-arg-end2-end: -2:14
  // macro-arg-expr-end: -3:13

  // CHECK4: "extracted(x, y)" [[@LINE-5]]:3 -> [[@LINE-5]]:13
  // CHECK5: "extracted(x)" [[@LINE-6]]:7 -> [[@LINE-6]]:13
}

// RUN: clang-refactor-test perform -action extract -selected=macro-arg-end1 %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test perform -action extract -selected=macro-arg-end2  %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test perform -action extract -selected=macro-arg-expr %s | FileCheck --check-prefix=CHECK5 %s
