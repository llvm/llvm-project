// RUN: %check_clang_tidy %s fuchsia-overloaded-operator %t

class A {
public:
  int operator+(int);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: overloading 'operator+' is disallowed
};

class B {
public:
  B &operator=(const B &Other);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:3: warning: overloading 'operator=' is disallowed
  B &operator=(B &&Other);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:3: warning: overloading 'operator=' is disallowed
};

A operator-(const A &A1, const A &A2);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: overloading 'operator-' is disallowed

void operator delete(void*, void*) throw();
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: overloading 'operator delete' is disallowed
