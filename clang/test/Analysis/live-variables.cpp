// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify %s
// expected-no-diagnostics
class B {
public:
  bool m;
  ~B() {} // The destructor ensures that the binary logical operator below is wrapped in the ExprWithCleanups.
};
B foo();
int getBool();
int *getPtr();
int test() {
  int r = 0;
  for (int x = 0; x< 10; x++) {
    int *p = getPtr();
    // Liveness info is not computed correctly due to the following expression.
    // This happens due to CFG being special cased for short circuit operators.
    // PR18159
    if (p != 0 && getBool() && foo().m && getBool()) {
      r = *p; // no warning
    }
  }
  return r;
}
