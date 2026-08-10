// RUN: %clangxx_msan %s -O0 -fsanitize-memory-use-after-dtor -o %t && %run %t
// RUN: %clangxx_msan %s -O1 -fsanitize-memory-use-after-dtor -o %t && %run %t
// RUN: %clangxx_msan %s -O2 -fsanitize-memory-use-after-dtor -o %t && %run %t
// RUN: %clangxx_msan %s -DCVPTR=1 -O2 -fsanitize-memory-use-after-dtor -fsanitize-memory-track-origins -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CVPTR
// RUN: %clangxx_msan %s -DEAVPTR=1 -O2 -fsanitize-memory-use-after-dtor -fsanitize-memory-track-origins -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=EAVPTR
// RUN: %clangxx_msan %s -DEDVPTR=1 -O2 -fsanitize-memory-use-after-dtor -fsanitize-memory-track-origins -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=EDVPTR

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
// CVPTR: Virtual table ptr was destroyed
// CVPTR: {{#0 0x.* in __sanitizer_dtor_callback_vptr}}
// CVPTR: {{#1 0x.* in ~C .*cpp:}}[[@LINE-28]]:
// CVPTR: {{#2 0x.* in main .*cpp:}}[[@LINE-7]]:
#endif

  // Multiple inheritance, so has multiple vtables
  E *e = new E();
  e->~E();
  // Both of these fail
#ifdef EAVPTR
  e->A_Foo();
// EAVPTR: Virtual table ptr was destroyed
// EAVPTR: {{#0 0x.* in __sanitizer_dtor_callback_vptr}}
// EAVPTR: {{#1 0x.* in ~E .*cpp:}}[[@LINE-25]]:
// EAVPTR: {{#2 0x.* in main .*cpp:}}[[@LINE-7]]:
#endif

#ifdef EDVPTR
  e->D_Foo();
// EDVPTR: Virtual table ptr was destroyed
// EDVPTR: {{#0 0x.* in __sanitizer_dtor_callback_vptr}}
// EDVPTR: {{#1 0x.* in ~E .*cpp:}}[[@LINE-33]]:
// EDVPTR: {{#2 0x.* in main .*cpp:}}[[@LINE-15]]:
#endif
}
