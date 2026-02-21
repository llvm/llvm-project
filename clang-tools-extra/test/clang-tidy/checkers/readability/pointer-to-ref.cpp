// RUN: %check_clang_tidy %s readability-pointer-to-ref %t

struct Foo {
  int X;
  void bar();
};

// Positive: always dereferenced, never null-checked.
void derefArrow(Foo *P) {
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: pointer parameter 'P' can be a reference
  P->bar();
  P->X = 42;
}

// Positive: dereference via unary operator.
int derefStar(int *P) {
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: pointer parameter 'P' can be a reference
  return *P + 1;
}

// Positive: multiple parameters, only the dereferenced one flagged.
void twoParams(int *A, int *B) {
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: pointer parameter 'A' can be a reference
  *A = 10;
  if (B)
    *B = 20;
}

// Negative: null-checked before use.
void nullChecked(Foo *P) {
  if (P)
    P->bar();
}

// Negative: compared to nullptr.
void comparedToNull(Foo *P) {
  if (P != nullptr)
    P->bar();
}

// Negative: passed to another function (used as raw pointer).
void takePtr(Foo *);
void passedAlong(Foo *P) {
  P->bar();
  takePtr(P);
}

// Negative: used as array (subscript).
void asArray(int *P) {
  P[0] = 1;
  P[1] = 2;
}

// Negative: void pointer (too generic).
void voidPtr(void *P) {
}

// Negative: unnamed parameter.
void unnamed(int *) {
}

// Negative: no dereference at all.
void noDereference(int *P) {
}

// Negative: parameter reassigned.
void reassigned(int *P) {
  int X = 0;
  P = &X;
  *P = 42;
}

// Negative: virtual method (signature must match base).
struct Base {
  virtual void vmethod(int *P);
};

// Negative: function pointer parameter.
void funcPtr(void (*P)(int)) {
  P(42);
}

// Negative: pointer arithmetic.
void arithmetic(int *P) {
  *(P + 1) = 0;
}
