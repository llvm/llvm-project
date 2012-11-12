// RUN: %clang_cc1 -fsyntax-only -triple x86_64-unknown-unknown %s -verify

typedef struct { unsigned long bits[(((1) + (64) - 1) / (64))]; } cpumask_t;
cpumask_t x;
void foo() {
  (void)x;
}
void bar() {
  char* a;
  double b;
  b = (double)a; // expected-error {{pointer cannot be cast to type}}
  a = (char*)b; // expected-error {{cannot be cast to a pointer type}}
}

long bar1(long *next) {
        return (long)(*next)++;  
}

typedef _Bool Bool;
typedef int Int;
typedef long Long;
typedef float Float;
typedef double Double;
typedef _Complex int CInt;
typedef _Complex long CLong;
typedef _Complex float CFloat;
typedef _Complex double CDouble;
typedef void *VoidPtr;
typedef char *CharPtr;

void testBool(Bool v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
  (void) (VoidPtr) v;
  (void) (CharPtr) v;
}

void testInt(Int v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
  (void) (VoidPtr) v; // expected-warning{{cast to 'VoidPtr' (aka 'void *') from smaller integer type 'Int' (aka 'int')}}
  (void) (CharPtr) v; // expected-warning{{cast to 'CharPtr' (aka 'char *') from smaller integer type 'Int' (aka 'int')}}
}

void testLong(Long v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
  (void) (VoidPtr) v;
  (void) (CharPtr) v;
}

void testFloat(Float v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testDouble(Double v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testCI(CInt v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testCLong(CLong v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testCFloat(CFloat v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testCDouble(CDouble v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (Float) v;
  (void) (Double) v;
  (void) (CInt) v;
  (void) (CLong) v;
  (void) (CFloat) v;
  (void) (CDouble) v;
}

void testVoidPtr(VoidPtr v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (VoidPtr) v;
  (void) (CharPtr) v;
}

void testCharPtr(CharPtr v) {
  (void) (Bool) v;
  (void) (Int) v;
  (void) (Long) v;
  (void) (VoidPtr) v;
  (void) (CharPtr) v;
}

typedef enum { x_a, x_b } X;
void *intToPointerCast2(X x) {
  return (void*)x;
}

void *intToPointerCast3() {
  return (void*)(1 + 3);
}
