// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep -e "global_ctors.*@A" %t
// RUN: grep -e "global_dtors.*@B" %t
// RUN: grep -e "global_ctors.*@C" %t
// RUN: grep -e "global_dtors.*@D" %t

int printf(const char *, ...);

void A() __attribute__((constructor));
void B() __attribute__((destructor));

void A() {
  printf("A\n");
}

void B() {
  printf("B\n");
}

static void C() __attribute__((constructor));

static void D() __attribute__((destructor));

static int foo() {
  return 10;
}

static void C() {
  printf("A: %d\n", foo());
}

static void D() {
  printf("B\n");
}

int main() {
  return 0;
}
