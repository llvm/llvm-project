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

// Positive: dereference through parentheses.
int derefParen(int *P) {
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: pointer parameter 'P' can be a reference
  return *(P) + 1;
}

// Positive: (*P)->member pattern.
struct Bar {
  int val;
};
int derefThenArrow(Bar **P) {
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: pointer parameter 'P' can be a reference
  return (*P)->val;
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

// Negative: compared using == nullptr.
void equalNull(Foo *P) {
  if (P == nullptr)
    return;
  P->bar();
}

// Negative: negation null check (!P).
void negationCheck(Foo *P) {
  if (!P)
    return;
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

// Negative: address-of usage.
void addressOf(int *P) {
  int **PP = &P;
  **PP = 42;
}

// Negative: extern "C" linkage.
extern "C" void cLinkage(int *P) {
  *P = 10;
}

// Negative: delete expression.
void deletePtr(Foo *P) {
  P->bar();
  delete P;
}

// Negative: return pointer value.
int *returnPtr(int *P) {
  *P = 10;
  return P;
}

// Negative: stored to another variable.
void storedToVar(int *P) {
  int *Q = P;
  *Q = 42;
}

// Negative: passed to constructor.
struct Wrapper {
  Wrapper(int *P);
};
void passedToCtor(int *P) {
  *P = 10;
  Wrapper W(P);
}

// Negative: used in sizeof (unevaluated context).
void sizeofUsage(int *P) {
  (void)sizeof(*P);
}

// Negative: sizeof with pointer itself.
void sizeofPtr(int *P) {
  (void)sizeof(P);
}

// Negative: alignof in unevaluated context.
void alignofUsage(int *P) {
  (void)alignof(decltype(*P));
}

// Negative: decltype (unevaluated context).
void decltypeUsage(int *P) {
  decltype(P) Q = nullptr;
  (void)Q;
}

// Negative: pointer compared to another pointer (not null).
void comparedToOtherPtr(int *P, int *Q) {
  if (P == Q)
    return;
  *P = 10;
}

// Negative: used in lambda capture by value (passed along).
void lambdaCapture(int *P) {
  *P = 10;
  auto F = [P]() { return *P; };
  (void)F;
}

// Negative: operator overload.
struct MyClass {
  bool operator==(const MyClass *Other) const;
};
