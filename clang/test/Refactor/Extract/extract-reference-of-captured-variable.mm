
#ifdef USECONST
#define CONST const
#else
#define CONST
#endif

typedef struct {
  int width, height;
} Rectangle;

@interface I

- (int)takesRef:(CONST int &)x;
+ (int)takesRef:(CONST int &)x;
- (int)takesVal:(int)x;
- (int)takesStructRef:(CONST Rectangle &)r;

@end

void methodTakesRef(I *i, int x, Rectangle r) {
  [i takesRef: x];
// CHECK1: extracted(I *i, int &x) {\nreturn [i takesRef: x];\n}
// CHECK2: extracted(I *i, int x) {\nreturn [i takesRef: x];\n}
  [I takesRef: x];
// CHECK1: extracted(int &x) {\nreturn [I takesRef: x];\n}
// CHECK2: extracted(int x) {\nreturn [I takesRef: x];\n}
  [i takesVal: x];
// CHECK1: extracted(I *i, int x) {\nreturn [i takesVal: x];\n}
// CHECK2: extracted(I *i, int x) {\nreturn [i takesVal: x];\n}
  [i takesStructRef: r];
// CHECK1: extracted(I *i, Rectangle &r) {\nreturn [i takesStructRef: r];\n}
// CHECK2: extracted(I *i, const Rectangle &r) {\nreturn [i takesStructRef: r];\n}
  [I takesRef: (r).width];
// CHECK1: extracted(Rectangle &r) {\nreturn [I takesRef: (r).width];\n}
// CHECK2: extracted(const Rectangle &r) {\nreturn [I takesRef: (r).width];\n}
}

class PrivateInstanceVariablesMethodCallRefs {
  int x;

  void methodTakesRef(I *j) {
    [j takesRef: x];
// CHECK1: extracted(I *j, int &x) {\nreturn [j takesRef: x];\n}
// CHECK2: extracted(I *j, int x) {\nreturn [j takesRef: x];\n}
  }
}

// RUN: clang-refactor-test perform -action extract -selected=%s:22:3-22:18 -selected=%s:25:3-25:18 -selected=%s:28:3-28:18 -selected=%s:31:3-31:24 -selected=%s:34:3-34:26 -selected=%s:43:5-43:20 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:22:3-22:18 -selected=%s:25:3-25:18 -selected=%s:28:3-28:18 -selected=%s:31:3-31:24 -selected=%s:34:3-34:26 -selected=%s:43:5-43:20 %s -DUSECONST | FileCheck --check-prefix=CHECK2 %s
