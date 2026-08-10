// RUN: %check_clang_tidy %s bugprone-parent-virtual-call %t

extern int foo();

class A {
public:
  A() = default;
  virtual ~A() = default;

  virtual int virt_1() { return foo() + 1; }
  virtual int virt_2() { return foo() + 2; }

  int non_virt() { return foo() + 3; }
  static int stat() { return foo() + 4; }
};

class B : public A {
public:
  B() = default;

  // Nothing to fix: calls to direct parent.
  int virt_1() override { return A::virt_1() + 3; }
  int virt_2() override { return A::virt_2() + 4; }
};

class C : public B {
public:
  int virt_1() override { return A::virt_1() + B::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' refers to a member overridden in subclass; did you mean 'B'? [bugprone-parent-virtual-call]
  // CHECK-FIXES:  int virt_1() override { return B::virt_1() + B::virt_1(); }
  int virt_2() override { return A::virt_1() + B::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'B'? {{.*}}
  // CHECK-FIXES:  int virt_2() override { return B::virt_1() + B::virt_1(); }

  // Test that non-virtual and static methods are not affected by this cherker.
  int method_c() { return A::stat() + A::non_virt(); }
};

// Check aliased type names
using A1 = A;
typedef A A2;
#define A3 A

class C2 : public B {
public:
  int virt_1() override { return A1::virt_1() + A2::virt_1() + A3::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A1::virt_1' {{.*}}; did you mean 'B'? {{.*}}
  // CHECK-MESSAGES: :[[@LINE-2]]:49: warning: qualified name 'A2::virt_1' {{.*}}; did you mean 'B'? {{.*}}
  // CHECK-MESSAGES: :[[@LINE-3]]:64: warning: qualified name 'A3::virt_1' {{.*}}; did you mean 'B'? {{.*}}
  // CHECK-FIXES:  int virt_1() override { return B::virt_1() + B::virt_1() + B::virt_1(); }
};

// Test that the check affects grand-grand..-parent calls too.
class D : public C {
public:
  int virt_1() override { return A::virt_1() + B::virt_1() + D::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'C'? {{.*}}
  // CHECK-MESSAGES: :[[@LINE-2]]:48: warning: qualified name 'B::virt_1' {{.*}}; did you mean 'C'? {{.*}}
  // CHECK-FIXES:  int virt_1() override { return C::virt_1() + C::virt_1() + D::virt_1(); }
  int virt_2() override { return A::virt_1() + B::virt_1() + D::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'C'? {{.*}}
  // CHECK-MESSAGES: :[[@LINE-2]]:48: warning: qualified name 'B::virt_1' {{.*}}; did you mean 'C'? {{.*}}
  // CHECK-FIXES:  int virt_2() override { return C::virt_1() + C::virt_1() + D::virt_1(); }
};

// Test classes in namespaces.
namespace {
class BN : public A {
public:
  int virt_1() override { return A::virt_1() + 3; }
  int virt_2() override { return A::virt_2() + 4; }
};
} // namespace

namespace N1 {
class A {
public:
  A() = default;
  virtual int virt_1() { return foo() + 1; }
  virtual int virt_2() { return foo() + 2; }
};
} // namespace N1

namespace N2 {
class BN : public N1::A {
public:
  int virt_1() override { return A::virt_1() + 3; }
  int virt_2() override { return A::virt_2() + 4; }
};
} // namespace N2

class CN : public BN {
public:
  int virt_1() override { return A::virt_1() + BN::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'BN'? {{.*}}
  // CHECK-FIXES:  int virt_1() override { return BN::virt_1() + BN::virt_1(); }
  int virt_2() override { return A::virt_1() + BN::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'BN'? {{.*}}
  // CHECK-FIXES:  int virt_2() override { return BN::virt_1() + BN::virt_1(); }
};

class CNN : public N2::BN {
public:
  int virt_1() override { return N1::A::virt_1() + N2::BN::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'N1::A::virt_1' {{.*}}; did you mean 'N2::BN'? {{.*}}
  // CHECK-FIXES:  int virt_1() override { return N2::BN::virt_1() + N2::BN::virt_1(); }
  int virt_2() override { return N1::A::virt_1() + N2::BN::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'N1::A::virt_1' {{.*}}; did you mean 'N2::BN'? {{.*}}
  // CHECK-FIXES:  int virt_2() override { return N2::BN::virt_1() + N2::BN::virt_1(); }
};

// Test multiple inheritance fixes
class AA {
public:
  AA() = default;
  virtual ~AA() = default;

  virtual int virt_1() { return foo() + 1; }
  virtual int virt_2() { return foo() + 2; }

  int non_virt() { return foo() + 3; }
  static int stat() { return foo() + 4; }
};

class BB_1 : virtual public AA {
public:
  BB_1() = default;

  // Nothing to fix: calls to parent.
  int virt_1() override { return AA::virt_1() + 3; }
  int virt_2() override { return AA::virt_2() + 4; }
};

class BB_2 : virtual public AA {
public:
  BB_2() = default;
  int virt_1() override { return AA::virt_1() + 3; }
};

class CC : public BB_1, public BB_2 {
public:
  int virt_1() override { return AA::virt_1() + 3; }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'AA::virt_1' refers to a member overridden in subclasses; did you mean 'BB_1' or 'BB_2'? {{.*}}
  // No fix available due to multiple choice of parent class.
};

// Test that virtual method is not diagnosed as not overridden in parent.
class BI : public A {
public:
  BI() = default;
};

class CI : BI {
  int virt_1() override { return A::virt_1(); }
};

// Test templated classes.
template <class F> class BF : public A {
public:
  int virt_1() override { return A::virt_1() + 3; }
};

// Test templated parent class.
class CF : public BF<int> {
public:
  int virt_1() override { return A::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'BF'? {{.*}}
};

// Test both templated class and its parent class.
template <class F> class DF : public BF<F> {
public:
  DF() = default;
  int virt_1() override { return A::virt_1(); }
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: qualified name 'A::virt_1' {{.*}}; did you mean 'BF'? {{.*}}
};

// Just to instantiate DF<F>.
int bar() { return (new DF<int>())->virt_1(); }
