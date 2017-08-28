// RUN: %clang_analyze_cc1 -Wno-unused-value -std=c++14 -analyzer-checker=core,debug.ExprInspection,alpha.core.PointerArithm -verify %s
struct X {
  int *p;
  int zero;
  void foo () {
    reset(p - 1);
  }
  void reset(int *in) {
    while (in != p) // Loop must be entered.
      zero = 1;
  }
};

int test (int *in) {
  X littleX;
  littleX.zero = 0;
  littleX.p = in;
  littleX.foo();
  return 5/littleX.zero; // no-warning
}


class Base {};
class Derived : public Base {};

void checkPolymorphicUse() {
  Derived d[10];

  Base *p = d;
  ++p; // expected-warning{{Pointer arithmetic on a pointer to base class is dangerous}}
}

void checkBitCasts() {
  long l;
  char *p = (char*)&l;
  p = p+2;
}

void checkBasicarithmetic(int i) {
  int t[10];
  int *p = t;
  ++p;
  int a = 5;
  p = &a;
  ++p; // expected-warning{{Pointer arithmetic on non-array variables relies on memory layout, which is dangerous}}
  p = p + 2; // expected-warning{{}}
  p = 2 + p; // expected-warning{{}}
  p += 2; // expected-warning{{}}
  a += p[2]; // expected-warning{{}}
  p = i*0 + p;
  p = p + i*0;
  p += i*0;
}

void checkArithOnSymbolic(int*p) {
  ++p;
  p = p + 2;
  p = 2 + p;
  p += 2;
  (void)p[2];
}

struct S {
  int t[10];
};

void arrayInStruct() {
  S s;
  int * p = s.t;
  ++p;
  S *sp = new S;
  p = sp->t;
  ++p;
  delete sp;
}

void checkNew() {
  int *p = new int;
  p[1] = 1; // expected-warning{{}}
}

void InitState(int* state) {
    state[1] = 1; // expected-warning{{}}
}

int* getArray(int size) {
    if (size == 0)
      return new int;
    return new int[5];
}

void checkConditionalArray() {
    int* maybeArray = getArray(0);
    InitState(maybeArray);
}

void checkMultiDimansionalArray() {
  int a[5][5];
   *(*(a+1)+2) = 2;
}

unsigned ptrSubtractionNoCrash(char *Begin, char *End) {
  auto N = End - Begin;
  if (Begin)
    return 0;
  return N;
}

// Bug 34309
bool ptrAsIntegerSubtractionNoCrash(__UINTPTR_TYPE__ x, char *p) {
  __UINTPTR_TYPE__ y = (__UINTPTR_TYPE__)p - 1;
  return y == x;
}
