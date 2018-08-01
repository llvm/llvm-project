
void takesInt(int x);

typedef struct { int width; int height; } Rectangle;

#ifdef USEINIT1
#define INIT1 r.width = 0; { int r, y; }
#else
#define INIT1 { int r, y; }
#endif

void extractCaptureUsedAfterSimple(int x) {
  Rectangle r;
  INIT1;
  int y = x * x;
#ifdef USEAFTER1
  takesInt(y);
#endif
#ifdef USEAFTER2
  takesInt(r.height);
#endif
#ifdef USEAFTER3
  r.width = 0;
#endif
#ifdef USEAFTER4
  y += 1;
#endif
}
// CHECK1: "static void extracted(int x) {\nRectangle r;\n  INIT1;\n  int y = x * x;\n}\n\n"
// CHECK1-NEXT: "extracted(x);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s | FileCheck --check-prefix=CHECK1 %s

// CHECK2: "static void extracted(Rectangle &r, int x, int &y) {\n\n  INIT1;\n  y = x * x;\n}\n\n"
// CHECK2-NEXT: "Rectangle r;\nint y;\nextracted(r, x, y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEINIT1 -DUSEAFTER1 -DUSEAFTER2 | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEINIT1 -DUSEAFTER3 -DUSEAFTER4 | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEINIT1 -DUSEAFTER1 -DUSEAFTER2 -DUSEAFTER3 -DUSEAFTER4 | FileCheck --check-prefix=CHECK2 %s

// CHECK3: "static void extracted(int x, int &y) {\n\n  INIT1;\n  y = x * x;\n}\n\n"
// CHECK3-NEXT: "Rectangle r;\nint y;\nextracted(x, y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEAFTER1 -DUSEAFTER2 | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEAFTER3 -DUSEAFTER4 | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEAFTER1 -DUSEAFTER2 -DUSEAFTER3 -DUSEAFTER4 | FileCheck --check-prefix=CHECK3 %s

// CHECK4: "static void extracted(int x, int &y) {\nRectangle r;\n  INIT1;\n  y = x * x;\n}\n\n"
// CHECK4-NEXT: "int y;\nextracted(x, y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEINIT1 -DUSEAFTER1 | FileCheck --check-prefix=CHECK4 %s

// CHECK5: "static void extracted(Rectangle &r, int x) {\n\n  INIT1;\n  int y = x * x;\n}\n\n"
// CHECK5-NEXT: "Rectangle r;\nextracted(r, x);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEINIT1 -DUSEAFTER3 | FileCheck --check-prefix=CHECK5 %s

// CHECK6: "static void extracted(int x) {\n\n  INIT1;\n  int y = x * x;\n}\n\n"
// CHECK6-NEXT: "Rectangle r;\nextracted(x);"
// RUN: clang-refactor-test perform -action extract -selected=%s:13:3-15:16 %s -DUSEAFTER3 | FileCheck --check-prefix=CHECK6 %s

void extractCaptureAfterUseMultipleDecls() {
#ifdef MULTIPLE_DECL1
  int x = 1, y = 2, z = 3; { int x, y, z; };
#endif
#ifdef MULTIPLE_DECL2
  int x, y = 2, *z; { int y; }
#endif
#ifdef MULTIPLE_DECL3
  int x = 1, y, * z = 0, a, b = {0}; { int a; }
#endif
#ifdef USEX
  x;
#endif
#ifdef USEY
  y;
#endif
#ifdef USEZ
  z;
#endif
#ifdef USEA
  a;
#endif
#ifdef USEB
  b;
#endif
}

// CHECK7: "static void extracted() {\nint x = 1, y = 2, z = 3;\n}\n\n"
// CHECK7-NEXT; "extracted();"
// RUN: clang-refactor-test perform -action extract -selected=%s:59:3-59:26 %s -DMULTIPLE_DECL1 | FileCheck --check-prefix=CHECK7 %s
// CHECK8: "static void extracted(int &x, int &y, int &z) {\nx = 1; y = 2; z = 3;\n}\n\n"
// CHECK8-NEXT: "int x;\nint y;\nint z;\nextracted(x, y, z);"
// RUN: clang-refactor-test perform -action extract -selected=%s:59:3-59:26 %s -DMULTIPLE_DECL1 -DUSEX -DUSEY -DUSEZ | FileCheck --check-prefix=CHECK8 %s
// CHECK9: "static void extracted(int &x) {\nx = 1; int y = 2; int z = 3; { int x, y, z; }\n}\n\n"
// CHECK9-NEXT: "int x;\nextracted(x);"
// RUN: clang-refactor-test perform -action extract -selected=%s:59:3-59:44 %s -DMULTIPLE_DECL1 -DUSEX | FileCheck --check-prefix=CHECK9 %s
// CHECK10: "static void extracted(int &y) {\nint x = 1; y = 2; int z = 3; { int x, y, z; }\n}\n\n"
// CHECK10-NEXT: "int y;\nextracted(y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:59:3-59:44 %s -DMULTIPLE_DECL1 -DUSEY | FileCheck --check-prefix=CHECK10 %s
// CHECK11: "static void extracted(int &z) {\nint x = 1; int y = 2; z = 3; { int x, y, z; }\n}\n\n"
// CHECK11-NEXT: "int z;\nextracted(z);"
// RUN: clang-refactor-test perform -action extract -selected=%s:59:3-59:44 %s -DMULTIPLE_DECL1 -DUSEZ | FileCheck --check-prefix=CHECK11 %s

// CHECK12: "static void extracted() {\nint x, y = 2, *z;\n}\n\n"
// CHECK12-NEXT: "extracted();"
// RUN: clang-refactor-test perform -action extract -selected=%s:62:3-62:19 %s -DMULTIPLE_DECL2 | FileCheck --check-prefix=CHECK12 %s
// CHECK13: "static void extracted(int &y) {\ny = 2;\n}\n\n"
// CHECK13-NEXT: "int x;\nint y;\nint * z;\nextracted(y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:62:3-62:19 %s -DMULTIPLE_DECL2 -DUSEX -DUSEY -DUSEZ | FileCheck --check-prefix=CHECK13 %s
// CHECK14: "static void extracted(int &y) {\nint x; y = 2; int * z; { int y; }\n}\n\n"
// CHECK14-NEXT: "int y;\nextracted(y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:62:3-62:31 %s -DMULTIPLE_DECL2 -DUSEY | FileCheck --check-prefix=CHECK14 %s

// CHECK15: "static void extracted() {\nint x = 1, y, * z = 0, a, b = {0};\n}\n\n"
// CHECK15-NEXT: "extracted();"
// RUN: clang-refactor-test perform -action extract -selected=%s:65:3-65:36 %s -DMULTIPLE_DECL3 | FileCheck --check-prefix=CHECK15 %s
// CHECK16: "static void extracted(int &x, int *&z) {\nx = 1; z = 0; int a; int b = {0};\n}\n\n"
// CHECK16-NEXT: "int x;\nint y;\nint * z;\nextracted(x, z);"
// RUN: clang-refactor-test perform -action extract -selected=%s:65:3-65:36 %s -DMULTIPLE_DECL3 -DUSEX -DUSEY -DUSEZ | FileCheck --check-prefix=CHECK16 %s
// CHECK17: "static void extracted(int &b, int &x, int *&z) {\nx = 1; z = 0; b = {0};\n}\n\n"
// CHECK17-NEXT: "int x;\nint y;\nint * z;\nint a;\nint b;\nextracted(b, x, z);"
// RUN: clang-refactor-test perform -action extract -selected=%s:65:3-65:36 %s -DMULTIPLE_DECL3 -DUSEX -DUSEY -DUSEZ -DUSEA -DUSEB | FileCheck --check-prefix=CHECK17 %s
// CHECK18: "static void extracted() {\nint x = 1; int y; int * z = 0; int b = {0}; { int a; }\n}\n\n"
// CHECK18-NEXT: "int a;\nextracted();"
// RUN: clang-refactor-test perform -action extract -selected=%s:65:3-65:48 %s -DMULTIPLE_DECL3 -DUSEA | FileCheck --check-prefix=CHECK18 %s

#define ONE 1

void preserveInitExpressionText() {
  int a = ONE;
  int x = ONE, y = ONE;
  a, y;
}

// CHECK19: "static void extracted(int &a, int &y) {\na = ONE;\n  int x = ONE; y = ONE;\n}\n\n
// CHECK19-NEXT: "int a;\nint y;\nextracted(a, y);"
// RUN: clang-refactor-test perform -action extract -selected=%s:126:3-127:23 %s | FileCheck --check-prefix=CHECK19 %s

class Construct {
public:
  Construct();
  Construct(int x, int y);
};

void handleConstruct() {
  Construct a(1, 2);
  Construct b(3, 4), c(5, 6);
  a, c;
}
// CHECK20: "static void extracted(Construct &a, Construct &c) {\na = Construct(1, 2);\n  Construct b(3, 4); c = Construct(5, 6);\n}\n\n"
// CHECK20-NEXT: "Construct a;\nConstruct c;\nextracted(a, c);"
// RUN: clang-refactor-test perform -action extract -selected=%s:142:3-143:29 %s | FileCheck --check-prefix=CHECK20 %s

struct Construct2 {
  int x, y;
};

void handleConstruct2() {
  Construct2 a = {1, 2};
  Construct2 b = {3, 4}, c = {5, 6};
  a, c;
}
// CHECK21: "static void extracted(Construct2 &a, Construct2 &c) {\na = {1, 2};\n  Construct2 b = {3, 4}; c = {5, 6};\n}\n\n"
// CHECK21-NEXT: "Construct2 a;\nConstruct2 c;\nextracted(a, c);"
// RUN: clang-refactor-test perform -action extract -selected=%s:155:3-156:29 %s -std=c++11 | FileCheck --check-prefix=CHECK21 %s

class Construct3 {
public:
  Construct3();
  Construct3(int x);
};

void handleConstruct3() {
  Construct3 a = 1;
  Construct3 b = 2, c = 3 + 3;
  a, c;
}
// CHECK22: "static void extracted(Construct3 &a, Construct3 &c) {\na = 1;\n  Construct3 b = 2; c = 3 + 3;\n}\n\n"
// CHECK22-NEXT: "Construct3 a;\nConstruct3 c;\nextracted(a, c);"
// RUN: clang-refactor-test perform -action extract -selected=%s:170:3-171:26 %s -std=c++11 | FileCheck --check-prefix=CHECK22 %s

void handleConstruct2InitList() {
  Construct2 a { 5, 6 };
  Construct2 b { 1, 2 }, c { 3, 4 };
  a, c;
}
// CHECK23: "static void extracted(Construct2 &a, Construct2 &c) {\na = { 5, 6 };\n  Construct2 b = { 1, 2 }; c = { 3, 4 };\n}\n\n"
// CHECK23-NEXT: "Construct2 a;\nConstruct2 c;\nextracted(a, c);"
// RUN: clang-refactor-test perform -action extract -selected=%s:179:3-180:36 %s -std=c++11 | FileCheck --check-prefix=CHECK23 %s

void onlyOneUsedAfterExtractionIsReturned() {
  const int x = 0;
  x;
  Construct a(1, 2);
  a;
}
// CHECK24: "static int extracted() {\nconst int x = 0;\nreturn x;\n}\n\n"
// CHECK24-NEXT: "const int x = extracted();"
// CHECK24: "static Construct extracted() {\nConstruct a(1, 2);\nreturn a;\n}\n\n"
// CHECK24-NEXT: "Construct a = extracted();"

int avoidReturningWhenReturnUsed() {
  int x = 0;
  if (x != 0) { return 22; }
  x;
}
// CHECK24: "static int extracted(int &x) {\nx = 0;\n  if (x != 0) { return 22; }\n}\n\n"
// CHECK24-NEXT: "int x;\nextracted(x);"

void returnOnlyWhenReturnReturnsNothing() {
  int x = 0;
  if (x != 0) { return; }
  x;
}
// CHECK24: "static int extracted() {\nint x = 0;\n  if (x != 0) { return x; }\nreturn x;\n}\n\n"
// CHECK24-NEXT: "int x = extracted();"

// RUN: clang-refactor-test perform -action extract -selected=%s:188:3-188:18 -selected=%s:190:3-190:20 -selected=%s:199:3-200:29 -selected=%s:207:3-208:26 %s -std=c++11 | FileCheck --check-prefix=CHECK24 %s
