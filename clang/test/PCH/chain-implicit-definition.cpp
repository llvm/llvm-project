// no PCH
// RUN: %clang_cc1 -emit-llvm-only -include %s -include %s %s
// with PCH
// RUN: %clang_cc1 -emit-llvm-only -chain-include %s -chain-include %s %s
#if !defined(PASS1)
#define PASS1

// A base with a virtual dtor.
struct A {
  virtual ~A();
};

// A derived class with an implicit virtual dtor.
struct B : A {
  // Key function to suppress vtable definition.
  virtual void virt();
};

#elif !defined(PASS2)
#define PASS2

// Further derived class that requires ~B().
// Causes definition of ~B(), but it was lost when saving PCH.
struct C : B {
  C();
  ~C() {}
};

#else

void foo() {
  // Variable that requires ~C().
  C c;
}

// VTable placement would again cause definition of ~B(), hiding the bug,
// if not for B::virt(), which suppresses the placement.

#endif
