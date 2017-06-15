
typedef struct {
  int width, height;
} Rectangle;

#ifdef ADD
#define MUT(x, y) x += y
#endif

#ifdef MUL
#define MUT(x, y) x *= y
#endif

#ifdef BIT
#define MUT(x, y) x |= y
#endif

#ifdef SHIFT
#define MUT(x, y) x >>= y
#endif

#ifdef INC1
#define MUT(x, y) ++x
#endif

#ifdef INC2
#define MUT(x, y) x++
#endif

#ifdef DEC1
#define MUT(x, y) --x
#endif

#ifdef DEC2
#define MUT(x, y) x--
#endif

#ifndef MUT
#define MUT(x, y) x = y
#endif

#ifdef FIELD
class MutatePrivateInstanceVariables {
  int x;
  int y;
  Rectangle r;

#endif

void mutateVariableOrField
#ifndef FIELD
  (int x, int y, Rectangle r)
#else
  ()
#endif
{
  (MUT(x, 1));
// CHECK1: (int &x) {\nreturn (MUT(x, 1));\n}

  (MUT((x), 1));
// CHECK1: (int &x) {\nreturn (MUT((x), 1));\n}

  (MUT(r.width, 1));
// CHECK1: (Rectangle &r) {\nreturn (MUT(r.width, 1));\n}

  (MUT((x, r.height), 1));
// CHECK1: (Rectangle &r, int x) {\nreturn (MUT((x, r.height), 1));\n}

  (MUT((x == 0 ? x : y), 1));
// CHECK1: (int &x, int &y) {\nreturn (MUT((x == 0 ? x : y), 1));\n}

  Rectangle a, b;
  (x == 0 ? (r) : b) = a;
// CHECK2: (const Rectangle &a, Rectangle &b, Rectangle &r, int x) {\nreturn (x == 0 ? (r) : b) = a;\n}

}

#ifdef FIELD
};
#endif

// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DFIELD | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test perform -action extract -selected=%s:73:3-73:25 %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:73:3-73:25 %s -DFIELD | FileCheck --check-prefix=CHECK2 %s

// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DADD | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DMUL -DFIELD | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DBIT | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DSHIFT -DFIELD | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DINC1 | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DINC2 -DFIELD | FileCheck --check-prefix=CHECK1 %s

// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DDEC1 | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:57:3-57:14 -selected=%s:60:3-60:16 -selected=%s:63:3-63:20 -selected=%s:66:3-66:26 -selected=%s:69:3-69:29 %s -DDEC2 -DFIELD | FileCheck --check-prefix=CHECK1 %s

void dontMutateVariable(int *array, int x) {
  array[x] = 0;
// CHECK3: (int *array, int x) {\narray[x] = 0;\n}
  *array = 0;
// CHECK3: (int *array) {\n*array = 0;\n}
  array = 0;
// CHECK3: extracted(int *&array) {\narray = 0;\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:101:3-101:15 -selected=%s:103:3-103:13 -selected=%s:105:3-105:12 %s | FileCheck --check-prefix=CHECK3 %s

#ifdef __cplusplus

int &returnsRef(int x) {
  static int result = 0;
  return result;
}

void dontMutateCallArguments(int x) {
  returnsRef(x) = 0;
// CHECK4: extracted(int x) {\nreturnsRef(x) = 0;\n}
}

void mutateRefVar(int &x) {
  x = 0;
// CHECK4: extracted(int &x) {\nx = 0;\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:119:3-119:20 -selected=%s:124:3-124:8 %s | FileCheck --check-prefix=CHECK4 %s

#endif
