// RUN: %clangxx -fno-lto -DIMPL -c -o %t.impl.o %s
// RUN: %clangxx_cfi -fdevirtualize-speculatively -o %t %s %t.impl.o
// RUN: %run %t

// -fdevirtualize-speculatively must not add local CFI checks to classes with
// public LTO visibility: their vtables may come from outside of the LTO unit,
// like the one defined in a non-LTO object here.

// FIXME: The checked load traps on the vtable from the non-LTO object.
// XFAIL: *

struct [[clang::lto_visibility_public]] A {
  virtual void f();
};

A *create();

#ifdef IMPL
void A::f() {}

struct B : A {
  void f() override {}
};

A *create() { return new B; }
#else
int main() { create()->f(); }
#endif
