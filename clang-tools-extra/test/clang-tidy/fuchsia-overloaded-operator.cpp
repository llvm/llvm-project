// RUN: %check_clang_tidy %s fuchsia-overloaded-operator %t

class A {
public:
  int operator+(int);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: cannot overload 'operator+' [fuchsia-overloaded-operator]
};

class B {
public:
  B &operator=(const B &Other);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:3: warning: cannot overload 'operator=' [fuchsia-overloaded-operator]
  B &operator=(B &&Other);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:3: warning: cannot overload 'operator=' [fuchsia-overloaded-operator]
};

A operator-(const A &A1, const A &A2);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: cannot overload 'operator-' [fuchsia-overloaded-operator]

void operator delete(void*, void*) throw();
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: cannot overload 'operator delete' [fuchsia-overloaded-operator]
