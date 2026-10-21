// Printing parsed ClangIR input with -emit-cir reproduces the CIR it was
// emitted as, including C++ constructs such as vtables, dynamic and
// thread-local initialization, and guarded static locals.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %t.cir -o %t2.cir
// RUN: diff %t.cir %t2.cir

struct Base {
  virtual ~Base();
  virtual int get() const { return 1; }
};

struct Derived : Base {
  int v;
  Derived(int v) : v(v) {}
  ~Derived() override {}
  int get() const override { return v; }
};

int compute();
int dynInit = compute();
thread_local int tls = 7;

int useStatic() {
  static Derived d(compute());
  return d.get() + tls;
}

Base *make() { return new Derived(3); }
