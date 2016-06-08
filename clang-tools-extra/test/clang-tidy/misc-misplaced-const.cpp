// RUN: %check_clang_tidy %s misc-misplaced-const %t

typedef int plain_i;
typedef int *ip;
typedef const int *cip;

void func() {
  if (const int *i = 0)
    ;
  if (const plain_i *i = 0)
    ;
  if (const cip i = 0)
    ;

  // CHECK-MESSAGES: :[[@LINE+1]]:16: warning: 'i' declared with a const-qualified typedef type; results in the type being 'int *const' instead of 'const int *'
  if (const ip i = 0)
    ;
}

template <typename Ty>
struct S {
  const Ty *i;
  const Ty &i2;
};

template struct S<int>;
template struct S<ip>; // ok
template struct S<cip>;

template <typename Ty>
struct U {
  const Ty *i;
  const Ty &i2;
};

template struct U<int *>; // ok

struct T {
  typedef void (T::*PMF)();

  void f() {
    const PMF val = &T::f; // ok
  }
};
