// RUN: %check_clang_tidy %s fuchsia-virtual-inheritance %t

class A {
public:
  A(int value) : val(value) {}

  int do_A() { return val; }

private:
  int val;
};

class B : public virtual A {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: direct virtual inheritance is disallowed [fuchsia-virtual-inheritance]
  // CHECK-NEXT: class B : public virtual A {
public:
  B() : A(0) {}
  int do_B() { return 1 + do_A(); }
};

class C : public virtual A {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: direct virtual inheritance is disallowed [fuchsia-virtual-inheritance]
  // CHECK-NEXT: class C : public virtual A {
public:
  C() : A(0) {}
  int do_C() { return 2 + do_A(); }
};

class D : public B, public C {
public:
  D(int value) : A(value), B(), C() {}

  int do_D() { return do_A() + do_B() + do_C(); }
};

int main() {
  A *a = new A(0);
  B *b = new B();
  C *c = new C();
  D *d = new D(0);
  return 0;
}
