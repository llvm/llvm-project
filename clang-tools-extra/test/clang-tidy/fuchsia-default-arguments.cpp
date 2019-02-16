// RUN: %check_clang_tidy %s fuchsia-default-arguments %t

int foo(int value = 5) { return value; }
// CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
// CHECK-FIXES: int foo(int value) { return value; }

int f() {
  foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments]
  // CHECK-NOTES: [[@LINE-7]]:9: note: default parameter was declared here
}

int bar(int value) { return value; }

int n() {
  foo(0);
  bar(0);
}

class Baz {
public:
  int a(int value = 5) { return value; }
  // CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
  // CHECK-FIXES: int a(int value) { return value; }

  int b(int value) { return value; }
};

class Foo {
  // Fix should be suggested in declaration
  int a(int value = 53);
  // CHECK-NOTES: [[@LINE-1]]:9: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
  // CHECK-FIXES: int a(int value);
};

// Fix shouldn't be suggested in implementation
int Foo::a(int value) {
  return value;
}

// Elided functions
void f(int = 5) {};
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
// CHECK-FIXES: void f(int) {};

void g(int) {};

// Should not suggest fix for macro-defined parameters
#define D(val) = val

void h(int i D(5));
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
// CHECK-FIXES-NOT: void h(int i);

void x(int i);
void x(int i = 12);
// CHECK-NOTES: [[@LINE-1]]:8: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
// CHECK-FIXES: void x(int i);

void x(int i) {}

struct S {
  void x(int i);
};

void S::x(int i = 12) {}
// CHECK-NOTES: [[@LINE-1]]:11: warning: declaring a parameter with a default argument is disallowed [fuchsia-default-arguments]
// CHECK-FIXES: void S::x(int i) {}

int main() {
  S s;
  s.x();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments]
  // CHECK-NOTES: [[@LINE-8]]:11: note: default parameter was declared here
  // CHECK-NEXT: void S::x(int i = 12) {}
  x();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments]
  // CHECK-NOTES: [[@LINE-18]]:8: note: default parameter was declared here
  // CHECK-NEXT: void x(int i = 12);
}
