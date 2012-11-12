// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -emit-llvm-only -verify %s -Wno-unreachable-code
// expected-no-diagnostics

int val = 42;
int& test1() {
  return throw val, val;
}

int test2() {
  return val ? throw val : val;
}

// rdar://problem/8608801
void test3() {
  throw false;
}

// PR10582
int test4() {
  return 1 ? throw val : val;
}
