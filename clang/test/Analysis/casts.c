// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -fenable-matrix -analyzer-checker=core,alpha.core,debug.ExprInspection -Wno-pointer-to-int-cast -Wno-strict-prototypes -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -fenable-matrix -analyzer-checker=core,alpha.core,debug.ExprInspection -Wno-pointer-to-int-cast -Wno-strict-prototypes -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -fenable-matrix -analyzer-checker=core,alpha.core,debug.ExprInspection -Wno-pointer-to-int-cast -Wno-strict-prototypes -verify -DEAGERLY_ASSUME=1 -w %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -fenable-matrix -analyzer-checker=core,alpha.core,debug.ExprInspection -Wno-pointer-to-int-cast -Wno-strict-prototypes -verify -DEAGERLY_ASSUME=1 -DBIT32=1 -w %s

extern void clang_analyzer_eval(_Bool);

// Test if the 'storage' region gets properly initialized after it is cast to
// 'struct sockaddr *'. 

typedef unsigned char __uint8_t;
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_socklen_t;
typedef __uint8_t sa_family_t;
typedef __darwin_socklen_t socklen_t;
struct sockaddr { sa_family_t sa_family; };
struct sockaddr_storage {};

void getsockname();

#ifndef EAGERLY_ASSUME

void f(int sock) {
  struct sockaddr_storage storage;
  struct sockaddr* sockaddr = (struct sockaddr*)&storage; // expected-warning{{Casting data to a larger structure type and accessing a field can lead to memory access errors or data corruption}}
  socklen_t addrlen = sizeof(storage);
  getsockname(sock, sockaddr, &addrlen);
  switch (sockaddr->sa_family) { // no-warning
  default:
    ;
  }
}

struct s {
  struct s *value;
};

void f1(struct s **pval) {
  int *tbool = ((void*)0);
  struct s *t = *pval;
  pval = &(t->value);
  tbool = (int *)pval; // use the cast-to type 'int *' to create element region.
  char c = (unsigned char) *tbool; // Should use cast-to type to create symbol.
  if (*tbool == -1) // here load the element region with the correct type 'int'
    (void)3;
}

void f2(const char *str) {
 unsigned char ch, cl, *p;

 p = (unsigned char *)str;
 ch = *p++; // use cast-to type 'unsigned char' to create element region.
 cl = *p++;
 if(!cl)
    cl = 'a';
}

// Test cast VariableSizeArray to pointer does not crash.
void *memcpy(void *, void const *, unsigned long);
typedef unsigned char Byte;
void doit(char *data, int len) {
    if (len) {
        Byte buf[len];
        memcpy(buf, data, len);
    }
}

// PR 6013 and 6035 - Test that a cast of a pointer to long and then to int does not crash SValuator.
void pr6013_6035_test(void *p) {
  unsigned int foo;
  foo = ((long)(p));
  (void) foo;
}

// PR12511 - Test that we support SymCastExpr, which represents symbolic int to float cast.
char ttt(int intSeconds) {
  double seconds = intSeconds;
  if (seconds)
    return 0;
  return 0;
}

int foo (int* p) {
  int y = 0;
  if (p == 0) {
    if ((*((void**)&p)) == (void*)0) // Test that the cast to void preserves the symbolic region.
      return 0;
    else
      return 5/y; // This code should be unreachable: no-warning.
  }
  return 0;
}

void castsToBool(void) {
  clang_analyzer_eval(0); // expected-warning{{FALSE}}
  clang_analyzer_eval(0U); // expected-warning{{FALSE}}
  clang_analyzer_eval((void *)0); // expected-warning{{FALSE}}

  clang_analyzer_eval(1); // expected-warning{{TRUE}}
  clang_analyzer_eval(1U); // expected-warning{{TRUE}}
  clang_analyzer_eval(-1); // expected-warning{{TRUE}}
  clang_analyzer_eval(0x100); // expected-warning{{TRUE}}
  clang_analyzer_eval(0x100U); // expected-warning{{TRUE}}
  clang_analyzer_eval((void *)0x100); // expected-warning{{TRUE}}

  extern int symbolicInt;
  clang_analyzer_eval(symbolicInt); // expected-warning{{UNKNOWN}}
  if (symbolicInt)
    clang_analyzer_eval(symbolicInt); // expected-warning{{TRUE}}

  extern void *symbolicPointer;
  clang_analyzer_eval(symbolicPointer); // expected-warning{{UNKNOWN}}
  if (symbolicPointer)
    clang_analyzer_eval(symbolicPointer); // expected-warning{{TRUE}}

  int localInt;
  int* ptr = &localInt;
  clang_analyzer_eval(ptr); // expected-warning{{TRUE}}
  clang_analyzer_eval(&castsToBool); // expected-warning{{TRUE}}
  clang_analyzer_eval("abc"); // expected-warning{{TRUE}}

  extern float globalFloat;
  clang_analyzer_eval(globalFloat); // expected-warning{{UNKNOWN}}
}

void locAsIntegerCasts(void *p) {
  int x = (int) p;
  clang_analyzer_eval(++x < 10); // no-crash // expected-warning{{UNKNOWN}}
}

void multiDimensionalArrayPointerCasts(void) {
  static int x[10][10];
  int *y1 = &(x[3][5]);
  char *z = ((char *) y1) + 2;
  int *y2 = (int *)(z - 2);
  int *y3 = ((int *)x) + 35; // This is offset for [3][5].

  clang_analyzer_eval(y1 == y2); // expected-warning{{TRUE}}

  // FIXME: should be FALSE (i.e. equal pointers).
  // FIXME: pointer subtraction warning might be incorrect
  clang_analyzer_eval(y1 - y2); // expected-warning{{UNKNOWN}}
  // expected-warning@-1{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  // FIXME: should be TRUE (i.e. same symbol).
  clang_analyzer_eval(*y1 == *y2); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(*((char *)y1) == *((char *) y2)); // expected-warning{{TRUE}}

  clang_analyzer_eval(y1 == y3); // expected-warning{{TRUE}}

  // FIXME: should be FALSE (i.e. equal pointers).
  // FIXME: pointer subtraction warning might be incorrect
  clang_analyzer_eval(y1 - y3); // expected-warning{{UNKNOWN}}
  // expected-warning@-1{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  // FIXME: should be TRUE (i.e. same symbol).
  clang_analyzer_eval(*y1 == *y3); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(*((char *)y1) == *((char *) y3)); // expected-warning{{TRUE}}
}

void *getVoidPtr(void);

void testCastVoidPtrToIntPtrThroughIntTypedAssignment(void) {
  int *x;
  (*((int *)(&x))) = (int)getVoidPtr();
  *x = 1; // no-crash
}

void testCastUIntPtrToIntPtrThroughIntTypedAssignment(void) {
  unsigned u;
  int *x;
  (*((int *)(&x))) = (int)&u;
  *x = 1;
  clang_analyzer_eval(u == 1); // expected-warning{{TRUE}}
}

void testCastVoidPtrToIntPtrThroughUIntTypedAssignment(void) {
  int *x;
  (*((int *)(&x))) = (int)(unsigned *)getVoidPtr();
  *x = 1; // no-crash
}

void testLocNonLocSymbolAssume(int a, int *b) {
  if ((int)b < a) {} // no-crash
}

void testLocNonLocSymbolRemainder(int a, int *b) {
  int c = ((int)b) % a;
  if (a == 1) {
    c += 1;
  }
}

void testSwitchWithSizeofs(void) {
  switch (sizeof(char) == 1) { // expected-warning{{switch condition has boolean value}}
  case sizeof(char):; // no-crash
  }
}

void test_ToUnion_cast(unsigned long long x) {
  union Key {
    unsigned long long data;
  };
  void clang_analyzer_dump_union(union Key);
  clang_analyzer_dump_union((union Key)x); // expected-warning {{Unknown}}
}

typedef char cx5x5 __attribute__((matrix_type(5, 5)));
typedef int ix5x5 __attribute__((matrix_type(5, 5)));
void test_MatrixCast_cast(cx5x5 c) {
  void clang_analyzer_dump_ix5x5(ix5x5);
  clang_analyzer_dump_ix5x5((ix5x5)c); // expected-warning {{Unknown}}
}

void test_VectorSplat_cast(long x) {
  typedef int __attribute__((ext_vector_type(2))) V;
  void clang_analyzer_dump_V(V);
  clang_analyzer_dump_V((V)x); // expected-warning {{Unknown}}
}

#endif

#ifdef EAGERLY_ASSUME

int globalA;
extern int globalFunc(void);
void no_crash_on_symsym_cast_to_long(void) {
  char c = globalFunc() - 5;
  c == 0;
  globalA -= c;
  globalA == 3;
  (long)globalA << 48;
  #ifdef BIT32
  // expected-warning@-2{{Left shift by '48' overflows the capacity of 'long'}}
  #else
  // expected-no-diagnostics
  #endif
}

#endif

char no_crash_SymbolCast_of_float_type_aux(int *p) {
  *p += 1;
  return *p;
}

void no_crash_SymbolCast_of_float_type(void) {
  extern float x;
  char (*f)() = no_crash_SymbolCast_of_float_type_aux;
  f(&x);
}

double no_crash_reinterpret_double_as_int(double a) {
  *(int *)&a = 1;
  return a * a;
}

double no_crash_reinterpret_double_as_ptr(double a) {
  *(void **)&a = 0;
  return a * a;
}

double no_crash_reinterpret_double_as_sym_int(double a, int b) {
  *(int *)&a = b;
  return a * a;
}

double no_crash_reinterpret_double_as_sym_ptr(double a, void * b) {
  *(void **)&a = b;
  return a * a;
}

void no_crash_reinterpret_char_as_uchar(char ***a, int *b) {
  *(unsigned char **)a = (unsigned char *)b;
  if (**a == 0) // no-crash
    ;
}

// PR50179.
struct S {};
void symbolic_offset(struct S *ptr, int i) {
  const struct S *pS = ptr + i;
  struct S s = *pS; // no-crash
}
