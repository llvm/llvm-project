// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

// RUN: %clangxx_msan %s -DCVPTR=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t

// RUN: %clangxx_msan %s -DEAVPTR=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t

// RUN: %clangxx_msan %s -DEDVPTR=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 not %run %t

// Expected to quit due to invalid access when invoking
// function using vtable.

class A {
 public:
  int x;
  virtual ~A() {
    // Should succeed
    this->A_Foo();
  }
  virtual void A_Foo() {}
};

class B : public virtual A {
 public:
  int y;
  virtual ~B() {}
  virtual void A_Foo() {}
};

class C : public B {
 public:
  int z;
  ~C() {}
};

class D {
 public:
  int w;
  ~D() {}
  virtual void D_Foo() {}
};

class E : public virtual A, public virtual D {
 public:
  int u;
  ~E() {}
  void A_Foo() {}
};

int main() {
  // Simple linear inheritance
  C *c = new C();
  c->~C();
  // This fails
#ifdef CVPTR
  c->A_Foo();
#endif

  // Multiple inheritance, so has multiple vtables
  E *e = new E();
  e->~E();
  // Both of these fail
#ifdef EAVPTR
  e->A_Foo();
#endif
#ifdef EDVPTR
  e->D_Foo();
#endif
}
