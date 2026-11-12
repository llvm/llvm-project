// RUN: %clangxx_asan -O0 %s -o %t

// RUN: not %run %t 'A' 2>&1 | FileCheck %s
// RUN: not %run %t 'B' 2>&1 | FileCheck %s

// Missing lifetime markers in test_a
// https://bugs.llvm.org/show_bug.cgi?id=34353
// XFAIL: *

struct B {
  B() : p('B') {}
  char p;
};

struct C {
  const char *p;
  explicit C(const char *c) : p(c) {}
  explicit C(const B &b) : p(&b.p) {}
};

struct A {
  char p;
  A() : p('C') {}
  const operator C() const { return C(&p); }
};

volatile char r;
void test_a() {
  C s = A();
  r = *s.p;
}

void test_b() {
  C s = B();
  r = *s.p;
}

int main(int argc, char **argv) {
  switch (argv[1][0]) {
  case 'A':
    test_a();
    return 0;
  case 'B':
    test_b();
    return 0;
  }
  return 1;
}

// CHECK: ERROR: AddressSanitizer: stack-use-after-scope
