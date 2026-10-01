// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-use-after-move %t

#include <utility>

struct A {
  int a;
};

struct B : A {
  int b;
  void f();
  B(B&& other) :
    A(std::move(other)),
    b(std::move(other.b))
    // CHECK-NOTES-NOT: [[@LINE-1]]:17: warning: 'other' used after it was moved
    {
      other.f();
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'other' used after it was moved
      // CHECK-NOTES: [[@LINE-6]]:5: note: move occurred here
    }
};

struct C : A {
  int c;
  C(C&& other) :
    A(std::move(other))
    {
      other.a;
      // CHECK-NOTES: [[@LINE-1]]:7: warning: 'other' used after it was moved
      // CHECK-NOTES: [[@LINE-4]]:5: note: move occurred here
    }
};

struct D { int d; };
struct E : A, D {
    E(E&& other) : A(std::move(other)) { other.d; }
};

