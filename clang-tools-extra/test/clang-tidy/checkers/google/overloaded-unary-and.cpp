// RUN: %check_clang_tidy %s google-runtime-operator %t

struct Foo {
  void *operator&();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not overload unary operator&, it is dangerous. [google-runtime-operator]
};

template <typename T>
struct TFoo {
  T *operator&();
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not overload unary operator&
};

TFoo<int> tfoo;

struct Bar;
void *operator&(Bar &b);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not overload unary operator&

// No warnings on binary operators.
struct Qux {
  void *operator&(Qux &q);
};

void *operator&(Qux &q, Qux &r);
