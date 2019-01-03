
struct Rectangle { int width, height; };

void extractStatement(const Rectangle &r) {
  int area = r.width * r.height;
// CHECK1: "static void extracted(const Rectangle &r) {\nint area = r.width * r.height;\n}\n\n"
// CHECK1-NEXT: "extracted(r);" [[@LINE-2]]:3 -> [[@LINE-2]]:33
  if (r.width) {
    int x = r.height;
  }
// CHECK1: "static void extracted(const Rectangle &r) {\nif (r.width) {\n    int x = r.height;\n  }\n}\n\n"
// CHECK1-NEXT: "extracted(r);" [[@LINE-4]]:3 -> [[@LINE-2]]:4
  if (r.width) {
    int x = r.height;
  } ; // This semicolon shouldn't be extracted.
// CHECK1: "static void extracted(const Rectangle &r) {\nif (r.width) {\n    int x = r.height;\n  }\n}\n\n"
// CHECK1-NEXT: "extracted(r);" [[@LINE-4]]:3 -> [[@LINE-2]]:4
  do {
  } while (true) ;
// CHECK1: "static void extracted() {\ndo {\n  } while (true) ;\n}\n\n"
// CHECK1-NEXT: "extracted();" [[@LINE-3]]:3 -> [[@LINE-2]]:19
  do {
  } while (true) /*we still want to take this semicolon*/ ;
// CHECK1: "static void extracted() {\ndo {\n  } while (true) /*we still want to take this semicolon*/ ;\n}\n\n"
// CHECK1-NEXT: "extracted();" [[@LINE-3]]:3 -> [[@LINE-2]]:60
}

// RUN: clang-refactor-test perform -action extract -selected=%s:5:3-5:32 -selected=%s:8:3-10:4 -selected=%s:13:3-15:4 -selected=%s:18:3-19:17 -selected=%s:22:3-23:17 %s | FileCheck --check-prefix=CHECK1 %s
;
void extractCantFindSemicolon() {
  do {
  } while (true)
  // Add a semicolon in both the extracted and original function as we don't
  // want to extract the semicolon below:
  ;
// CHECK2: "static void extracted() {\ndo {\n  } while (true);\n}\n\n"
// CHECK2-NEXT: "extracted();" [[@LINE-6]]:3 -> [[@LINE-5]]:17
}
// RUN: clang-refactor-test perform -action extract -selected=%s:31:3-32:17 %s | FileCheck --check-prefix=CHECK2 %s

void extractedStmtNoNeedForSemicolon() {
  {
    int x = 0;
  }
// CHECK3: "static void extracted() {\n{\n    int x = 0;\n  }\n}\n\n"
  switch (2) {
  case 1:
    break;
  case 2:
    break;
  }
// CHECK3: "static void extracted() {\nswitch (2) {\n  case 1:\n    break;\n  case 2:\n    break;\n  }\n}\n\n"
  while (true) {
    int x = 0;
  }
// CHECK3: "static void extracted() {\nwhile (true) {\n    int x = 0;\n  }\n}\n\n"
  for (int i = 0; i < 10; ++i) {
  }
// CHECK3: "static void extracted() {\nfor (int i = 0; i < 10; ++i) {\n  }\n}\n\n"
  struct XS {
    int *begin() { return 0; }
    int *end() { return 0; }
  };
  XS xs;
  for (int i : xs) {
  }
// CHECK3: "static void extracted(const XS &xs) {\nfor (int i : xs) {\n  }\n}\n\n"
  try { int x = 0; }
  catch (const int &i) {
    int y = i;
  }
// CHECK3: "static void extracted() {\ntry { int x = 0; }\n  catch (const int &i) {\n    int y = i;\n  }\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:42:3-44:4 -selected=%s:46:3-51:4 -selected=%s:53:3-55:4 -selected=%s:57:3-58:4 -selected=%s:65:3-66:4 -selected=%s:68:3-71:4 %s -std=c++11 | FileCheck --check-prefix=CHECK3 %s
;
void extractStatementRange(int x) {
  x = 2;
  int y = 0;
  extractStatementRange(x);
  if (x == 2) {
    int z = 0;
  }
  x = 2;

// CHECK4: "static void extracted(int x) {\nextractStatementRange(x);\n  if (x == 2) {\n    int z = 0;\n  }\n}\n\n" [[@LINE-9]]:1
// CHECK4-NEXT: "extracted(x);" [[@LINE-7]]:3 -> [[@LINE-4]]:4

// CHECK4: "static void extracted(int x) {\nint y = 0;\n  extractStatementRange(x);\n}\n\n" [[@LINE-12]]:1
// CHECK4-NEXT: "extracted(x)" [[@LINE-11]]:3 -> [[@LINE-10]]:27

// CHECK4: "static void extracted(int &x) {\nx = 2;\n  int y = 0;\n  extractStatementRange(x);\n}\n\n" [[@LINE-15]]:1
// CHECK4-NEXT: "extracted(x)" [[@LINE-15]]:3 -> [[@LINE-13]]:27

// CHECK4: "static void extracted(int &x) {\nx = 2;\n  int y = 0;\n}\n\n" [[@LINE-18]]:1
// CHECK4-NEXT: "extracted(x);" [[@LINE-18]]:3 -> [[@LINE-17]]:13
}

// RUN: clang-refactor-test perform -action extract -selected=%s:80:3-83:4 -selected=%s:79:3-80:6 -selected=%s:78:3-80:6 -selected=%s:78:3-79:6 %s | FileCheck --check-prefix=CHECK4 %s

void extractedVariableUsedAndDefinedInExtractedCode(int x) {
  int y = x;
  if (y == 1) {
    int z = 0;
  }
// CHECK5: "static void extracted(int x) {\nint y = x;\n  if (y == 1) {\n    int z = 0;\n  }\n}\n\n"
// CHECK5-NEXT: "extracted(x);"
// CHECK5: "static void extracted(int y) {\nif (y == 1) {\n    int z = 0;\n  }\n}\n\n"
// CHECK5-NEXT: "extracted(y);"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:102:2-105:4 -selected=%s:103:2-105:4 %s | FileCheck --check-prefix=CHECK5 %s

void extractAssignmentAsStatementOrExpr(int x) {
  x = 2;
// CHECK6: "static void extracted(int &x) {\nx = 2;\n}\n\n"
  x = x = 3;
// CHECK6: "static int extracted(int &x) {\nreturn x = 3;\n}\n\n"
  (void)(x = 4);
// CHECK6: "static int extracted(int &x) {\nreturn x = 4;\n}\n\n"
  if (x = 5) {
  }
// CHECK6: "static int extracted(int &x) {\nreturn x = 5;\n}\n\n"
  if (true)
    x = 6;
// CHECK6: "static void extracted(int &x) {\nx = 6;\n}\n\n"
  bool b = 2;
  if (b = false) {
  }
// CHECK6: "static bool extracted(bool &b) {\nreturn b = false;\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:115:3-115:8 -selected=%s:117:7-117:12 -selected=%s:119:10-119:15 -selected=%s:121:7-121:12 -selected=%s:125:5-125:10 -selected=%s:128:7-128:16 %s | FileCheck --check-prefix=CHECK6 %s

void extractCompoundAssignmentAsStatementOrExpr(int x) {
  x += 2;
// CHECK7: "static void extracted(int &x) {\nx += 2;\n}\n\n"
  x = x += 3;
// CHECK7: "static int extracted(int &x) {\nreturn x += 3;\n}\n\n"
  if (x *= 4) {
  }
// CHECK7: "static int extracted(int &x) {\nreturn x *= 4;\n}\n\n"
}

// RUN: clang-refactor-test perform -action extract -selected=%s:136:3-136:9 -selected=%s:138:7-138:13 -selected=%s:140:7-140:13 %s | FileCheck --check-prefix=CHECK7 %s

int inferReturnTypeFromReturnStatement(int x) {
  if (x == 0) {
    return x;
  }
  if (x == 1) {
    return x + 1;
  }
  return x + 2;
}
// CHECK8: "static int extracted(int x) {\nif (x == 1) {\n    return x + 1;\n  }\n  return x + 2;\n}\n\n"
// CHECK8: "static int extracted(int x) {\nif (x == 1) {\n    return x + 1;\n  }\n}\n\n"

// RUN: clang-refactor-test perform -action extract -selected=%s:151:3-154:15 -selected=%s:151:3-153:4 %s | FileCheck --check-prefix=CHECK8 %s

void careForNonCompoundSemicolons() {
// if-open-begin:+1:1
  if (true)
    careForNonCompoundSemicolons();
// if-open-end: -1:35
// CHECK9: "static void extracted() {\nif (true)\n    careForNonCompoundSemicolons();\n}\n\n"
// CHECK9: "extracted();" [[@LINE-4]]:3 -> [[@LINE-3]]:36

// for-open-begin:+1:1
  for (int i = 0; i < 10; ++i)
    while (i != 0)
      ;
// for-open-end: +0:1
// CHECK9:"static void extracted() {\nfor (int i = 0; i < 10; ++i)\n    while (i != 0)\n      ;\n}\n\n" [[@LINE-15]]:1
// CHECK9: "extracted();" [[@LINE-5]]:3 -> [[@LINE-3]]:8
}

// RUN: clang-refactor-test perform -action extract -selected=if-open -selected=for-open %s | FileCheck --check-prefix=CHECK9 %s
