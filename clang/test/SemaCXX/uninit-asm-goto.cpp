// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++11 -Wuninitialized -verify %s

// test1: Expect no diagnostics
int test1(int x) {
    int y;
    asm goto("" : "=r"(y) : "r"(x) : : err);
    return y;
  err:
    return -1;
}

// test2: Expect no diagnostics
int test2(int x) {
  int y;
  if (x < 42)
    asm goto("" : "+S"(x), "+D"(y) : "r"(x) :: indirect_1, indirect_2);
  else
    asm goto("" : "+S"(x), "+D"(y) : "r"(x), "r"(y) :: indirect_1, indirect_2);
  return x + y;
indirect_1:
  return -42;
indirect_2:
  return y;
}

// test3: Expect no diagnostics
int test3(int x) {
  int y;
  asm goto("" : "=&r"(y) : "r"(x) : : fail);
normal:
  y += x;
  return y;
  if (x) {
fail:
    return y;
  }
  return 0;
}

// test4: Expect no diagnostics
int test4(int x) {
  int y;
  goto forward;
backward:
  return y;
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

// test7: Expect no diagnostics.
int test7(int z) {
    int x;
    if (z)
      asm goto ("":"=r"(x):::A1,A2);
    return 0;
    A1:
    A2:
    return x;
}

// test8: Expect no diagnostics
int test8() {
    int x = 0;
    asm goto ("":"=r"(x):::A1,A2);
    return 0;
    A1:
    A2:
    return x;
}

// test9: Expect no diagnostics
int test9 (int x) {
    int y;
    asm goto("": "=r"(y) :::out);
    return 42;
out:
    return y;
}

int test10() {
  int y; // expected-note {{initialize the variable 'y' to silence this warning}}
  asm goto(""::::out);
  return 42;
out:
  return y; // expected-warning {{variable 'y' is uninitialized when used here}}
}
