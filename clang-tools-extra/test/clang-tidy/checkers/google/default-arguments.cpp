// RUN: %check_clang_tidy %s google-default-arguments %t

struct A {
  virtual void f(int I, int J = 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: default arguments on virtual or override methods are prohibited [google-default-arguments]
};

struct B : public A {
  void f(int I, int J = 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: default arguments on virtual or override methods are prohibited
};

struct C : public B {
  void f(int I, int J = 5) override;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: default arguments on virtual or override methods are prohibited
};

// Negatives.
struct D : public B {
  void f(int I, int J) override;
};

struct X {
  void f(int I, int J = 3);
};

struct Y : public X {
  void f(int I, int J = 5);
};
