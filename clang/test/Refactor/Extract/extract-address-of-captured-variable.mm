
void takesVoidPtr(void *x) { }
void takesPtr(int *x) { }
void takesPtrPtr(int **x) { }

#ifdef USECONST
#define CONST const
#else
#define CONST
#endif

typedef struct {
  int width, height;
} Rectangle;

@interface I

- (int)takesPtr:(CONST int *)x;
+ (int)takesPtr:(CONST int *)x;
- (int)takesVoidPtr:(CONST void *)x;
- (int)takesStructPtr:(CONST Rectangle *)r;

@end

void methodTakesPtr(I *i, int x, Rectangle r) {
  [i takesPtr: &x];
// CHECK1: extracted(I *i, int &x) {\nreturn [i takesPtr: &x];\n}
// CHECK2: extracted(I *i, const int &x) {\nreturn [i takesPtr: &x];\n}
  [I takesPtr: (&x)];
// CHECK1: extracted(int &x) {\nreturn [I takesPtr: (&x)];\n}
// CHECK2: extracted(const int &x) {\nreturn [I takesPtr: (&x)];\n}
  [i takesVoidPtr: (&(x))];
// CHECK1: extracted(I *i, int &x) {\nreturn [i takesVoidPtr: (&(x))];\n}
// CHECK2: extracted(I *i, const int &x) {\nreturn [i takesVoidPtr: (&(x))];\n}
  [i takesStructPtr: &r];
// CHECK1: extracted(I *i, Rectangle &r) {\nreturn [i takesStructPtr: &r];\n}
// CHECK2: extracted(I *i, const Rectangle &r) {\nreturn [i takesStructPtr: &r];\n}
  [I takesPtr: &(r).width];
// CHECK1: extracted(Rectangle &r) {\nreturn [I takesPtr: &(r).width];\n}
// CHECK2: extracted(const Rectangle &r) {\nreturn [I takesPtr: &(r).width];\n}
}

// RUN: clang-refactor-test perform -action extract -selected=%s:26:3-26:19 -selected=%s:29:3-29:21 -selected=%s:32:3-32:27 -selected=%s:35:3-35:25 -selected=%s:38:3-38:27 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action extract -selected=%s:26:3-26:19 -selected=%s:29:3-29:21 -selected=%s:32:3-32:27 -selected=%s:35:3-35:25 -selected=%s:38:3-38:27 %s -DUSECONST | FileCheck --check-prefix=CHECK2 %s
