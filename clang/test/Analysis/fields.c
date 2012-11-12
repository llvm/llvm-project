// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core,debug.ExprInspection %s -analyzer-store=region -verify

void clang_analyzer_eval(int);

unsigned foo();
typedef struct bf { unsigned x:2; } bf;
void bar() {
  bf y;
  *(unsigned*)&y = foo();
  y.x = 1;
}

struct s {
  int n;
};

void f() {
  struct s a;
  int *p = &(a.n) + 1;
}

typedef struct {
  int x,y;
} Point;

Point getit(void);
void test() {
  Point p;
  (void)(p = getit()).x;
}


void testLazyCompoundVal() {
  Point p = {42, 0};
  Point q;
  clang_analyzer_eval((q = p).x == 42); // expected-warning{{TRUE}}
  clang_analyzer_eval(q.x == 42); // expected-warning{{TRUE}}
}
