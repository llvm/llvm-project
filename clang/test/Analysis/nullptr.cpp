// RUN: %clang_analyze_cc1 -std=c++11 -Wno-conversion-null -analyzer-checker=core,debug.ExprInspection -analyzer-store region -analyzer-output=text -verify %s

void clang_analyzer_eval(int);

// test to see if nullptr is detected as a null pointer
void foo1(void) {
  char  *np = nullptr; // expected-note{{'np' initialized to a null pointer value}}
  *np = 0;  // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
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
  char *np = nullptr; // expected-note{{'np' initialized to a null pointer value}}
  // casting a nullptr to anything should be caught eventually
  int *ip = &(((struct foo *)np)->f); // expected-note{{'ip' initialized to a null pointer value}}
  *ip = 0;  // expected-warning{{Dereference of null pointer}}
            // expected-note@-1{{Dereference of null pointer}}
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
  char **p = 0; // expected-note{{'p' initialized to a null pointer value}}
  delete *(p + 0); // expected-warning{{Dereference of null pointer}}
                   // expected-note@-1{{Dereference of null pointer}}
}

void zoo1backwards() {
  char **p = 0; // expected-note{{'p' initialized to a null pointer value}}
  delete *(0 + p); // expected-warning{{Dereference of null pointer}}
                   // expected-note@-1{{Dereference of null pointer}}
}

typedef __INTPTR_TYPE__ intptr_t;
void zoo1multiply() {
  char **p = 0; // expected-note{{'p' initialized to a null pointer value}}
  delete *((char **)((intptr_t)p * 2)); // expected-warning{{Dereference of null pointer}}
                   // expected-note@-1{{Dereference of null pointer}}
}

void zoo2() {
  int **a = 0;
  int **b = 0; // expected-note{{'b' initialized to a null pointer value}}
  asm ("nop"
      :"=r"(*a)
      :"0"(*b) // expected-warning{{Dereference of null pointer}}
               // expected-note@-1{{Dereference of null pointer}}
      );
}

int exprWithCleanups() {
  struct S {
    S(int a):a(a){}
    ~S() {}

    int a;
  };

  int *x = 0; // expected-note{{'x' initialized to a null pointer value}}
  return S(*x).a; // expected-warning{{Dereference of null pointer}}
                  // expected-note@-1{{Dereference of null pointer}}
}

int materializeTempExpr() {
  int *n = 0; // expected-note{{'n' initialized to a null pointer value}}
  struct S {
    int a;
    S(int i): a(i) {}
  };
  const S &s = S(*n); // expected-warning{{Dereference of null pointer}}
                      // expected-note@-1{{Dereference of null pointer}}
  return s.a;
}

typedef decltype(nullptr) nullptr_t;
void testMaterializeTemporaryExprWithNullPtr() {
  // Create MaterializeTemporaryExpr with a nullptr inside.
  const nullptr_t &r = nullptr;
}

int getSymbol();

struct X {
  virtual void f() {}
};

void invokeF(X* x) {
  x->f(); // expected-warning{{Called C++ object pointer is null}}
          // expected-note@-1{{Called C++ object pointer is null}}
}

struct Type {
  decltype(nullptr) x;
};

void shouldNotCrash() {
  decltype(nullptr) p; // expected-note{{'p' declared without an initial value}}
  if (getSymbol()) // expected-note   {{Assuming the condition is false}}
                   // expected-note@-1{{Taking false branch}}
                   // expected-note@-2{{Assuming the condition is true}}
                   // expected-note@-3{{Taking true branch}}
    invokeF(p);    // expected-note   {{Calling 'invokeF'}}
                   // expected-note@-1{{Passing null pointer value via 1st parameter 'x'}}
  if (getSymbol()) {  // expected-note  {{Assuming the condition is true}}
                      // expected-note@-1{{Taking true branch}}
    X *xx = Type().x; // expected-note   {{Null pointer value stored to field 'x'}}
                      // expected-note@-1{{'xx' initialized to a null pointer value}}
    xx->f(); // expected-warning{{Called C++ object pointer is null}}
            // expected-note@-1{{Called C++ object pointer is null}}
  }
}

void f(decltype(nullptr) p) {
  int *q = nullptr;
  clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(q == 0); // expected-warning{{TRUE}}
                               // expected-note@-1{{TRUE}}
}

decltype(nullptr) returnsNullPtrType();
void fromReturnType() {
  ((X *)returnsNullPtrType())->f(); // expected-warning{{Called C++ object pointer is null}}
                                    // expected-note@-1{{Called C++ object pointer is null}}
}

#define AS_ATTRIBUTE __attribute__((address_space(256)))
class AS1 {
public:
  int x;
  ~AS1() {
    int AS_ATTRIBUTE *x = 0;
    *x = 3; // no-warning
  }
};
void test_address_space_field_access() {
  AS1 AS_ATTRIBUTE *pa = 0;
  pa->x = 0; // no-warning
}
void test_address_space_bind() {
  AS1 AS_ATTRIBUTE *pa = 0;
  AS1 AS_ATTRIBUTE &r = *pa;
  r.x = 0; // no-warning
}
