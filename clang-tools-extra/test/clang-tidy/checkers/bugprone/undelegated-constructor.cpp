// RUN: %check_clang_tidy %s bugprone-undelegated-constructor %t

struct Ctor;
Ctor foo();

struct Ctor {
  Ctor();
  Ctor(int);
  Ctor(int, int);
  Ctor(Ctor *i) {
    Ctor();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor? A temporary object is created here instead [bugprone-undelegated-constructor]
    Ctor(0);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor?
    Ctor(1, 2);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor?
    foo();
  }
};

Ctor::Ctor() {
  Ctor(1);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: did you intend to call a delegated constructor?
}

Ctor::Ctor(int i) : Ctor(i, 1) {} // properly delegated.

struct Dtor {
  Dtor();
  Dtor(int);
  Dtor(int, int);
  Dtor(Ctor *i) {
    Dtor();
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor?
    Dtor(0);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor?
    Dtor(1, 2);
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: did you intend to call a delegated constructor?
  }
  ~Dtor();
};

struct Base {};
struct Derived : public Base {
  Derived() { Base(); }
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: did you intend to call a delegated constructor?
};

template <typename T>
struct TDerived : public Base {
  TDerived() { Base(); }
};

TDerived<int> t;
