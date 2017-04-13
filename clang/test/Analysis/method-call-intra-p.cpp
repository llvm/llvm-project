// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store region -verify %s
// expected-no-diagnostics

// Intra-procedural C++ tests.

// Test relaxing function call arguments invalidation to be aware of const
// arguments. radar://10595327
struct InvalidateArgs {
  void ttt(const int &nptr);
  virtual void vttt(const int *nptr);
};
struct ChildOfInvalidateArgs: public InvalidateArgs {
  virtual void vttt(const int *nptr);
};
void declarationFun(int x) {
  InvalidateArgs t;
  x = 3;
  int y = x + 1;
  int *p = 0;
  t.ttt(y);
  if (x == y)
      y = *p; // no-warning
}
void virtualFun(int x) {
  ChildOfInvalidateArgs t;
  InvalidateArgs *pt = &t;
  x = 3;
  int y = x + 1;
  int *p = 0;
  pt->vttt(&y);
  if (x == y)
      y = *p; // no-warning
}
