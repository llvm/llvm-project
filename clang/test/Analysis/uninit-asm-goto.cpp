// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++11 -Wuninitialized -verify %s

// test1: Expect no diagnostics
int test1(int x) {
    int y;
    asm goto("nop" : "=r"(y) : "r"(x) : : err);
    return y;
  err:
    return -1;
}

int test2(int x) {
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}} \
         // expected-note {{initialize the variable}}
  if (x < 42)
    asm volatile goto("testl %0, %0; testl %1, %2; jne %l3" : "+S"(x), "+D"(y) : "r"(x) :: indirect_1, indirect_2);
  else
    asm volatile goto("testl %0, %1; testl %2, %3; jne %l5" : "+S"(x), "+D"(y) : "r"(x), "r"(y) :: indirect_1, indirect_2);
  return x + y;
indirect_1:
  return -42;
indirect_2:
  return y; // expected-note {{uninitialized use occurs here}}
}

int test3(int x) {
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}} \
         // expected-note {{initialize the variable}}
  asm goto("xorl %1, %0; jmp %l2" : "=&r"(y) : "r"(x) : : fail);
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
  int y; // expected-warning {{variable 'y' is used uninitialized whenever its declaration is reached}} \
         // expected-note {{initialize the variable}}
  goto forward;
backward:
  return y; // expected-note {{uninitialized use occurs here}}
forward:
  asm goto("# %0 %1 %2" : "=r"(y) : "r"(x) : : backward);
  return y;
}

// test5: Expect no diagnostics
int test5(int x) {
  int y;
  asm volatile goto("testl %0, %0; testl %1, %2; jne %l3" : "+S"(x), "+D"(y) : "r"(x) :: indirect, fallthrough);
fallthrough:
  return y;
indirect:
  return -2;
}
