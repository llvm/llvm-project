// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-use-after-move %t

#include <utility>

struct A {
  int a;
};

struct B : A {
  int b;
  B(B&& other) :
    A(std::move(other)),
    b(std::move(other.b)) // No error raised
        {}
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
