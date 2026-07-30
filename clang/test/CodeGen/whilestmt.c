// RUN: %clang_cc1 %s -emit-llvm -o -

int bar(void);
int foo(void) {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    i = bar();
  };
  return i;
}


int foo1(void) {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      break;
    i = bar();
  };
  return i;
}


int foo2(void) {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      continue;
    i = bar();
  };
  return i;
}


int foo3(void) {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      break;
  };
  return i;
}


int foo4(void) {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      continue;
  };
  return i;
}
