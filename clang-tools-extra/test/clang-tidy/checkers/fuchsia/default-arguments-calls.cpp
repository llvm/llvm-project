// RUN: %check_clang_tidy %s fuchsia-default-arguments-calls %t

int foo(int value = 5) { return value; }

int f() {
  foo();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments-calls]
  // CHECK-NOTES: [[@LINE-5]]:9: note: default parameter was declared here
}

int bar(int value) { return value; }

int n() {
  foo(0);
  bar(0);
}

void x(int i = 12);

struct S {
  void x(int i);
};

void S::x(int i = 12) {}

int main() {
  S s;
  s.x();
  // CHECK-NOTES: [[@LINE-1]]:5: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments-calls]
  // CHECK-NOTES: [[@LINE-6]]:11: note: default parameter was declared here
  // CHECK-NEXT: void S::x(int i = 12) {}
  x();
  // CHECK-NOTES: [[@LINE-1]]:3: warning: calling a function that uses a default argument is disallowed [fuchsia-default-arguments-calls]
  // CHECK-NOTES: [[@LINE-16]]:8: note: default parameter was declared here
  // CHECK-NEXT: void x(int i = 12);
}
