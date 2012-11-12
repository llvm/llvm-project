// RUN: %clang_cc1 -std=c++11 -Wno-conversion-null -analyze -analyzer-checker=core -analyzer-store region -verify %s

// test to see if nullptr is detected as a null pointer
void foo1(void) {
  char  *np = nullptr;
  *np = 0;  // expected-warning{{Dereference of null pointer}}
}

// check if comparing nullptr to nullptr is detected properly
void foo2(void) {
  char *np1 = nullptr;
  char *np2 = np1;
  char c;
  if (np1 == np2)
    np1 = &c;
  *np1 = 0;  // no-warning
}

// invoving a nullptr in a more complex operation should be cause a warning
void foo3(void) {
  struct foo {
    int a, f;
  };
  char *np = nullptr;
  // casting a nullptr to anything should be caught eventually
  int *ip = &(((struct foo *)np)->f);
  *ip = 0;  // expected-warning{{Dereference of null pointer}}
  // should be error here too, but analysis gets stopped
//  *np = 0;
}

// nullptr is implemented as a zero integer value, so should be able to compare
void foo4(void) {
  char *np = nullptr;
  if (np != 0)
    *np = 0;  // no-warning
  char  *cp = 0;
  if (np != cp)
    *np = 0;  // no-warning
}

int pr10372(void *& x) {
  // GNU null is a pointer-sized integer, not a pointer.
  x = __null;
  // This used to crash.
  return __null;
}

void zoo1() {
  char **p = 0;
  delete *(p + 0); // expected-warning{{Dereference of null pointer}}
}

void zoo2() {
  int **a = 0;
  int **b = 0;
  asm ("nop"
      :"=r"(*a)
      :"0"(*b) // expected-warning{{Dereference of null pointer}}
      );
}

int exprWithCleanups() {
  struct S {
    S(int a):a(a){}
    ~S() {}

    int a;
  };

  int *x = 0;
  return S(*x).a; // expected-warning{{Dereference of null pointer}}
}

int materializeTempExpr() {
  int *n = 0;
  struct S {
    int a;
    S(int i): a(i) {}
  };
  const S &s = S(*n); // expected-warning{{Dereference of null pointer}}
  return s.a;
}
