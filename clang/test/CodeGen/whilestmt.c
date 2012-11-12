// RUN: %clang_cc1 %s -emit-llvm -o -

int bar();
int foo() {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    i = bar();
  };
  return i;
}


int foo1() {
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


int foo2() {
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


int foo3() {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      break;
  };
  return i;
}


int foo4() {
  int i;
  i = 1 + 2;
  while(1) {
    i = bar();
    if (i == 42)
      continue;
  };
  return i;
}
