// RUN: %clangxx_cfi -o %t %s
// RUN: %run %t

// In this example, both __typeid_A_global_addr and __typeid_B_global_addr will
// refer to the same address. Make sure that the compiler does not assume that
// they do not alias.

struct A {
  virtual void f() = 0;
};

struct B : A {
  virtual void f() {}
};

__attribute__((weak)) void foo(void *p) {
  B *b = (B *)p;
  A *a = (A *)b;
  a->f();
}

int main() {
  B b;
  foo(&b);
}
