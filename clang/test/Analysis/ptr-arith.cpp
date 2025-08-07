// RUN: %clang_analyze_cc1 -Wno-unused-value -std=c++14 -verify %s -triple x86_64-pc-linux-gnu \
// RUN:    -analyzer-checker=core,debug.ExprInspection,alpha.core.PointerArithm

// RUN: %clang_analyze_cc1 -Wno-unused-value -std=c++14 -verify %s -triple x86_64-pc-linux-gnu \
// RUN:    -analyzer-config support-symbolic-integer-casts=true \
// RUN:    -analyzer-checker=core,debug.ExprInspection,alpha.core.PointerArithm

template <typename T> void clang_analyzer_dump(T);

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

// Bug 34374
bool integerAsPtrSubtractionNoCrash(char *p, __UINTPTR_TYPE__ m) {
  auto n = p - reinterpret_cast<char*>((__UINTPTR_TYPE__)1);
  return n == m;
}

namespace Bug_55934 {
struct header {
  unsigned a : 1;
  unsigned b : 1;
};
struct parse_t {
  unsigned bits0 : 1;
  unsigned bits2 : 2; // <-- header
  unsigned bits4 : 4;
};
int parse(parse_t *p) {
  unsigned copy = p->bits2;
  clang_analyzer_dump(copy);
  // expected-warning@-1 {{reg_$2<unsigned int Element{SymRegion{reg_$0<parse_t * p>},0 S64b,struct Bug_55934::parse_t}.bits2>}}
  header *bits = (header *)&copy;
  clang_analyzer_dump(bits->b);
  // expected-warning@-1 {{derived_$4{reg_$2<unsigned int Element{SymRegion{reg_$0<parse_t * p>},0 S64b,struct Bug_55934::parse_t}.bits2>,Element{copy,0 S64b,struct Bug_55934::header}.b}}}
  return bits->b; // no-warning
}
} // namespace Bug_55934

void LValueToRValueBitCast_dumps(void *p, char (*array)[8]) {
  clang_analyzer_dump(p);
  clang_analyzer_dump(array);
  // expected-warning@-2 {{&SymRegion{reg_$0<void * p>}}}
  // expected-warning@-2 {{&SymRegion{reg_$1<char (*)[8] array>}}}
  clang_analyzer_dump((unsigned long)p);
  clang_analyzer_dump(__builtin_bit_cast(unsigned long, p));
  // expected-warning@-2 {{&SymRegion{reg_$0<void * p>} [as 64 bit integer]}}
  // expected-warning@-2 {{&SymRegion{reg_$0<void * p>} [as 64 bit integer]}}
  clang_analyzer_dump((unsigned long)array);
  clang_analyzer_dump(__builtin_bit_cast(unsigned long, array));
  // expected-warning@-2 {{&SymRegion{reg_$1<char (*)[8] array>} [as 64 bit integer]}}
  // expected-warning@-2 {{&SymRegion{reg_$1<char (*)[8] array>} [as 64 bit integer]}}
}

unsigned long ptr_arithmetic(void *p) {
  return __builtin_bit_cast(unsigned long, p) + 1; // no-crash
}
