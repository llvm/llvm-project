// RUN: %check_clang_tidy %s misc-override-with-different-visibility %t -- -config="{CheckOptions: {misc-override-with-different-visibility.CheckDestructors: true,misc-override-with-different-visibility.CheckOperators: true}}" -- -I %S/Inputs/override-with-different-visibility
#include <test-system-header.h>
class A {
public:
  virtual void pub_foo1() {}
  virtual void pub_foo2() {}
  virtual void pub_foo3() {}
protected:
  virtual void prot_foo1();
  virtual void prot_foo2();
  virtual void prot_foo3();
private:
  virtual void priv_foo1() {}
  virtual void priv_foo2() {}
  virtual void priv_foo3() {}
};

void A::prot_foo1() {}
void A::prot_foo2() {}
void A::prot_foo3() {}

namespace test1 {

class B: public A {
public:
  void pub_foo1() override {}
  void prot_foo1() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo1' is changed from protected in class 'A' to public [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :9:16: note: function declared here as protected
  void priv_foo1() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo1' is changed from private in class 'A' to public [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :13:16: note: function declared here as private
protected:
  void pub_foo2() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'pub_foo2' is changed from public in class 'A' to protected [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :6:16: note: function declared here as public
  void prot_foo2() override {}
  void priv_foo2() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo2' is changed from private in class 'A' to protected [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :14:16: note: function declared here as private
private:
  void pub_foo3() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'pub_foo3' is changed from public in class 'A' to private [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :7:16: note: function declared here as public
  void prot_foo3() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo3' is changed from protected in class 'A' to private [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :11:16: note: function declared here as protected
  void priv_foo3() override {}
};

class C: public B {
public:
  void pub_foo1() override;
protected:
  void prot_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo1' is changed from public in class 'B' to protected [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :27:8: note: function declared here as public
private:
  void priv_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo1' is changed from public in class 'B' to private [misc-override-with-different-visibility]
  // CHECK-MESSAGES: :30:8: note: function declared here as public
};

void C::prot_foo1() {}
void C::priv_foo1() {}

}

namespace test2 {

class B: public A {
public:
  void pub_foo1() override;
protected:
  void prot_foo1() override;
private:
  void priv_foo1() override;
};

class C: public B {
public:
  void pub_foo1() override;
  void prot_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo1' is changed from protected in class 'B' to public
  // CHECK-MESSAGES: :75:8: note: function declared here as protected
  void priv_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo1' is changed from private in class 'B' to public
  // CHECK-MESSAGES: :77:8: note: function declared here as private

  void pub_foo2() override;
  void prot_foo2() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo2' is changed from protected in class 'A' to public
  // CHECK-MESSAGES: :10:16: note: function declared here as protected
  void priv_foo2() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo2' is changed from private in class 'A' to public
  // CHECK-MESSAGES: :14:16: note: function declared here as private
};

}

namespace test3 {

class B: private A {
public:
  void pub_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'pub_foo1' is changed from private (through private inheritance of class 'A') to public
  // CHECK-MESSAGES: :103:10: note: 'A' is inherited as private here
  // CHECK-MESSAGES: :5:16: note: function declared here as public
protected:
  void prot_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo1' is changed from private (through private inheritance of class 'A') to protected
  // CHECK-MESSAGES: :103:10: note: 'A' is inherited as private here
  // CHECK-MESSAGES: :9:16: note: function declared here as protected
private:
  void priv_foo1() override;

public:
  void prot_foo2() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'prot_foo2' is changed from private (through private inheritance of class 'A') to public
  // CHECK-MESSAGES: :103:10: note: 'A' is inherited as private here
  // CHECK-MESSAGES: :10:16: note: function declared here as protected
  void priv_foo2() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'priv_foo2' is changed from private in class 'A' to public
  // CHECK-MESSAGES: :14:16: note: function declared here as private

private:
  void pub_foo3() override;
  void prot_foo3() override;
};

class C: private A {
};

class D: public C {
public:
  void pub_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'pub_foo1' is changed from private (through private inheritance of class 'A') to public
  // CHECK-MESSAGES: :131:10: note: 'A' is inherited as private here
  // CHECK-MESSAGES: :5:16: note: function declared here as public
};


}

namespace test4 {

struct Base1 {
public:
  virtual void foo1();
private:
  virtual void foo2();
};

struct Base2 {
public:
  virtual void foo2();
private:
  virtual void foo1();
};

struct A : public Base1, public Base2 {
protected:
  void foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'foo1' is changed from private in class 'Base2' to protected
  // CHECK-MESSAGES: :158:16: note: function declared here as private
  // CHECK-MESSAGES: :[[@LINE-3]]:8: warning: visibility of function 'foo1' is changed from public in class 'Base1' to protected
  // CHECK-MESSAGES: :149:16: note: function declared here as public
private:
  void foo2() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'foo2' is changed from public in class 'Base2' to private
  // CHECK-MESSAGES: :156:16: note: function declared here as public
};

}

namespace test5 {

struct B1: virtual public A {};
struct B2: virtual private A {};
struct B: public B1, public B2 {
public:
  void pub_foo1() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'pub_foo1' is changed from private (through private inheritance of class 'A') to public
  // CHECK-MESSAGES: :179:12: note: 'A' is inherited as private here
  // CHECK-MESSAGES: :5:16: note: function declared here as public
};

}

namespace test_using {

class A {
private:
  A(int);
protected:
  virtual void f();
};

class B: public A {
public:
  using A::A;
  using A::f;
};

}

namespace test_template {

template <typename T>
class A {
protected:
  virtual T foo();
};

template <typename T>
class B: public A<T> {
private:
  T foo() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: visibility of function 'foo' is changed from protected in class 'A<int>' to private
  // CHECK-MESSAGES: :[[@LINE-8]]:13: note: function declared here as protected
};

template <typename T>
class C: private A<T> {
public:
  T foo() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: visibility of function 'foo' is changed from private (through private inheritance of class 'A<int>') to public
  // CHECK-MESSAGES: :[[@LINE-4]]:10: note: 'A<int>' is inherited as private here
  // CHECK-MESSAGES: :[[@LINE-17]]:13: note: function declared here as protected
};

B<int> fB() {
  return B<int>{};
}

C<int> fC() {
  return C<int>{};
}

}

namespace test_system_header {

struct SysDerived: public sys::Base {
private:
  void publicF();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: visibility of function 'publicF' is changed from public in class 'Base' to private
};

}

namespace test_destructor {

class A {
public:
  virtual ~A();
};

class B: public A {
protected:
  ~B();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: visibility of function '~B'
  // CHECK-MESSAGES: :[[@LINE-7]]:11: note: function declared here
};

}

namespace test_operator {

class A {
  virtual int operator()(int);
  virtual A& operator++();
  virtual operator double() const;
};

class B: public A {
protected:
  int operator()(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: visibility of function 'operator()'
  // CHECK-MESSAGES: :[[@LINE-9]]:15: note: function declared here
  A& operator++();
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: visibility of function 'operator++'
  // CHECK-MESSAGES: :[[@LINE-11]]:14: note: function declared here
  operator double() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: visibility of function 'operator double'
  // CHECK-MESSAGES: :[[@LINE-13]]:11: note: function declared here
};

}
