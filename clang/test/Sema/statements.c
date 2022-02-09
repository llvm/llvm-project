// RUN: %clang_cc1 %s -fsyntax-only -verify  -triple x86_64-pc-linux-gnu -Wno-unevaluated-expression

typedef unsigned __uint32_t;

#define __byte_swap_int_var(x) \
__extension__ ({ register __uint32_t __X = (x); \
   __asm ("bswap %0" : "+r" (__X)); \
   __X; })

int test(int _x) {
 return (__byte_swap_int_var(_x));
}

// PR2374
int test2(void) { return ({L:5;}); }
int test3(void) { return ({ {5;} }); }         // expected-error {{returning 'void' from a function with incompatible result type 'int'}}\
                                           // expected-warning {{expression result unused}}
int test4(void) { return ({ ({5;}); }); }
int test5(void) { return ({L1: L2: L3: 5;}); }
int test6(void) { return ({5;}); }
void test7(void) { ({5;}); }                   // expected-warning {{expression result unused}}

// PR3062
int test8[({10;})]; // expected-error {{statement expression not allowed at file scope}}

// PR3912
void test9(const void *P) {
  __builtin_prefetch(P);
}


void *test10(void) { 
bar:
  return &&bar;  // expected-warning {{returning address of label, which is local}}
}

// PR38569: Don't warn when returning a label from a statement expression.
void test10_logpc(void*);
void test10a(void) {
  test10_logpc(({
    my_pc:
      &&my_pc;
  }));
}

// PR6034
void test11(int bit) {
  switch (bit)
  switch (env->fpscr)  // expected-error {{use of undeclared identifier 'env'}}
  {
  }
}

// rdar://3271964
enum Numbers { kOne,  kTwo,  kThree,  kFour};
int test12(enum Numbers num) {
  switch (num == kOne) {// expected-warning {{switch condition has boolean value}}
  default: 
  case kThree:
    break;
  }
}


enum x { a, b, c, d, e, f, g };

void foo(enum x X) {
  switch (X) { // expected-warning {{enumeration value 'g' not handled in switch}}
  case a:
  case b:
  case c:
  case d:
  case e:
  case f:
    break;
  }

  switch (X) { // expected-warning {{enumeration values 'f' and 'g' not handled in switch}}
  case a:
  case b:
  case c:
  case d:
  case e:
    break;
  }

  switch (X) {  // expected-warning {{enumeration values 'e', 'f', and 'g' not handled in switch}}
    case a:
    case b:
    case c:
    case d:
      break;
  }

  switch (X) { // expected-warning {{5 enumeration values not handled in switch: 'c', 'd', 'e'...}}
  case a:
  case b:
    break;
  }
}

int test_pr8880(void) {
  int first = 1;
  for ( ; ({ if (first) { first = 0; continue; } 0; }); )
    return 0;
  return 1;
}

// In PR22849, we considered __ptr to be a static data member of the anonymous
// union. Now we declare it in the parent DeclContext.
void test_pr22849(void) {
  struct Bug {
    typeof(({ unsigned long __ptr; (int *)(0); })) __val;
    union Nested {
      typeof(({ unsigned long __ptr; (int *)(0); })) __val;
    } n;
  };
  enum E {
    SIZE = sizeof(({unsigned long __ptr; __ptr;}))
  };
}

// GCC ignores empty statements at the end of compound expressions where the
// result type is concerned.
void test13(void) {
  int a;
  a = ({ 1; });
  a = ({1;; });
  a = ({int x = 1; (void)x; }); // expected-error {{assigning to 'int' from incompatible type 'void'}}
  a = ({int x = 1; (void)x;; }); // expected-error {{assigning to 'int' from incompatible type 'void'}}
}

void test14(void) { return ({}); }
void test15(void) {
  return ({;;;; });
}
void test16(void) {
  return ({test:;; });
}
