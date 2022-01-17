// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++11 -Wuninitialized -verify %s

// test1: Expect no diagnostics
int test1(int x) {
    int y;
    asm goto("" : "=r"(y) : "r"(x) : : err);
    return y;
  err:
    return -1;
}

int test2(int x) {
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}}
         // expected-note@-1 {{initialize the variable}}
  if (x < 42)
    asm goto("" : "+S"(x), "+D"(y) : "r"(x) :: indirect_1, indirect_2);
  else
    asm goto("" : "+S"(x), "+D"(y) : "r"(x), "r"(y) :: indirect_1, indirect_2);
  return x + y;
indirect_1:
  return -42;
indirect_2:
  return y; // expected-note {{uninitialized use occurs here}}
}

int test3(int x) {
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}}
         // expected-note@-1 {{initialize the variable}}
  asm goto("" : "=&r"(y) : "r"(x) : : fail);
normal:
  y += x;
  return y;
  if (x) {
fail:
    return y; // expected-note {{uninitialized use occurs here}}
  }
  return 0;
}

int test4(int x) {
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}}
         // expected-note@-1 {{initialize the variable}}
  goto forward;
backward:
  return y; // expected-note {{uninitialized use occurs here}}
forward:
  asm goto("" : "=r"(y) : "r"(x) : : backward);
  return y;
}

// test5: Expect no diagnostics
int test5(int x) {
  int y;
  asm goto("" : "+S"(x), "+D"(y) : "r"(x) :: indirect, fallthrough);
fallthrough:
  return y;
indirect:
  return -2;
}

// test6: Expect no diagnostics.
int test6(unsigned int *x) {
  unsigned int val;

  // See through casts and unary operators.
  asm goto("" : "=r" (*(unsigned int *)(&val)) ::: indirect);
  *x = val;
  return 0;
indirect:
  return -1;
}

int test7(int z) {
    int x; // expected-warning {{variable 'x' is used uninitialized whenever its declaration is reached}}
           // expected-note@-1 {{initialize the variable 'x' to silence this warning}}
    if (z)
      asm goto ("":"=r"(x):::A1,A2);
    return 0;
    A1:
    A2:
    return x; // expected-note {{uninitialized use occurs here}}
}

int test8() {
    int x = 0; // expected-warning {{variable 'x' is used uninitialized whenever its declaration is reached}}
               // expected-note@-1 {{variable 'x' is declared here}}
    asm goto ("":"=r"(x):::A1,A2);
    return 0;
    A1:
    A2:
    return x; // expected-note {{uninitialized use occurs here}}
}
