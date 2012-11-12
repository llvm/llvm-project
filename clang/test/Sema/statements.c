// RUN: %clang_cc1 %s -fsyntax-only -verify

typedef unsigned __uint32_t;

#define __byte_swap_int_var(x) \
__extension__ ({ register __uint32_t __X = (x); \
   __asm ("bswap %0" : "+r" (__X)); \
   __X; })

int test(int _x) {
 return (__byte_swap_int_var(_x));
}

// PR2374
int test2() { return ({L:5;}); }
int test3() { return ({ {5;} }); }         // expected-error {{returning 'void' from a function with incompatible result type 'int'}}\
                                           // expected-warning {{expression result unused}}
int test4() { return ({ ({5;}); }); }
int test5() { return ({L1: L2: L3: 5;}); }
int test6() { return ({5;}); }
void test7() { ({5;}); }                   // expected-warning {{expression result unused}}

// PR3062
int test8[({10;})]; // expected-error {{statement expression not allowed at file scope}}

// PR3912
void test9(const void *P) {
  __builtin_prefetch(P);
}


void *test10() { 
bar:
  return &&bar;  // expected-warning {{returning address of label, which is local}}
}

// PR6034
void test11(int bit) {
  switch (bit) // expected-warning {{switch statement has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}
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

// PR 8880
// FIXME: Clang should reject this, since GCC does.  Previously this
// was causing a crash in the CFG builder.
int test_pr8880() {
  int first = 1;
  for ( ; ({ if (first) { first = 0; continue; } 0; }); )
    return 0;
  return 1;
}

