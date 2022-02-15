// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int bar(void);
int test0(void) {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    i = bar();
  } while(0);
  return i;
}


int test1(void) {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      break;
    i = bar();
  } while(1);
  return i;
}


int test2(void) {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      continue;
    i = bar();
  } while(1);
  return i;
}


int test3(void) {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      break;
  } while(0);
  return i;
}


int test4(void) {
  int i;
  i = 1 + 2;
  do {
    i = bar();
    if (i == 42)
      continue;
  } while(0);
  return i;
}

// rdar://6103124
void test5(void) {
  do { break; } while(0);
}

// PR14191
void test6f(void);
void test6(void) {
  do {
  } while (test6f(), 0);
  // CHECK: call {{.*}}void @test6f()
}

