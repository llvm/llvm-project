// RUN: %check_clang_tidy %s fuchsia-default-arguments-declarations %t

int foo(int value = 5) { return value; }
// CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
// CHECK-FIXES: int foo(int value) { return value; }

int bar(int value) { return value; }

class Baz {
public:
  int a(int value = 5) { return value; }
  // CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
  // CHECK-FIXES: int a(int value) { return value; }

  int b(int value) { return value; }
};

class Foo {
  // Fix should be suggested in declaration
  int a(int value = 53);
  // CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
  // CHECK-FIXES: int a(int value);
};

// Fix shouldn't be suggested in implementation
int Foo::a(int value) {
  return value;
}

// Elided functions
void f(int = 5) {};
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
// CHECK-FIXES: void f(int) {};

void g(int) {};

// Should not suggest fix for macro-defined parameters
#define D(val) = val

void h(int i D(5));
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
// CHECK-FIXES-NOT: void h(int i);

void x(int i);
void x(int i = 12);
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
// CHECK-FIXES: void x(int i);

void x(int i) {}

struct S {
  void x(int i);
};

void S::x(int i = 12) {}
// CHECK-NOTES: [[@LINE-1]]:11: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments-declarations]
// CHECK-FIXES: void S::x(int i) {}
