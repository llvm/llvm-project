#define ONE 1

void extractExprMacros(int x) {
  bool b = x == ONE;
// CHECK1: {\nreturn x == ONE;\n}
  int y = ONE + x;
// CHECK1: {\nreturn ONE + x;\n}
  int z = ONE - ONE * ONE;
// CHECK1: {\nreturn ONE - ONE * ONE;\n}
// CHECK1: {\nreturn ONE * ONE;\n}
// CHECK1: {\nreturn - ONE * ONE;\n}
}
// RUN: clang-refactor-test perform -action extract -selected=%s:4:12-4:20 -selected=%s:6:11-6:19 -selected=%s:8:12-8:26 -selected=%s:8:17-8:26 -selected=%s:8:15-8:26 %s | FileCheck --check-prefix=CHECK1 %s

#define MACRO2(x, y) x, y

int function(int x);

// MACRO-ARG3: "static int extracted(int x, int &y, int z) {\nreturn y = MACRO2(x + 2, function(z));\n}\n\n" [[@LINE+4]]:1
// MACRO-ARG2: "static int extracted(int z) {\nreturn function(z);\n}\n\n" [[@LINE+3]]:1
// MACRO-ARG1: "static int extracted(int x) {\nreturn x + 2;\n}\n\n" [[@LINE+2]]:1
;
void extractFromMacroArgument(int x, int y, int z) {

  // macro-arg-expr4-begin: +4:7
  // macro-arg-expr3-begin: +3:14
  // macro-arg-expr2-begin: +2:21
  // macro-arg-expr1-begin: +1:14
  y = MACRO2(x + 2, function(z)); // comment4
  // macro-arg-expr1-end: -1:19 // MACRO-ARG1: "extracted(x)" [[@LINE-1]]:14 -> [[@LINE-1]]:19
  // macro-arg-expr2-end: -2:32 // MACRO-ARG2: "extracted(z)" [[@LINE-2]]:21 -> [[@LINE-2]]:32
  // macro-arg-expr3-end: -3:32 // MACRO-ARG3: "extracted(x, y, z)" [[@LINE-3]]:3 -> [[@LINE-3]]:33
  // macro-arg-expr4-end: -4:19
}

// RUN: clang-refactor-test perform -action extract -selected=macro-arg-expr1 %s | FileCheck --check-prefix=MACRO-ARG1 %s
// RUN: clang-refactor-test perform -action extract -selected=macro-arg-expr2 %s | FileCheck --check-prefix=MACRO-ARG2 %s
// RUN: clang-refactor-test perform -action extract -selected=macro-arg-expr3 %s | FileCheck --check-prefix=MACRO-ARG3 %s
// RUN: clang-refactor-test perform -action extract -selected=macro-arg-expr4 %s | FileCheck --check-prefix=MACRO-ARG3 %s
