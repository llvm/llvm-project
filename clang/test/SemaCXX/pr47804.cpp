// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify

template <class InputIt, class Pred>
bool all_of(InputIt first, Pred p);

template <typename T> void load_test() {
  // Ensure that this doesn't crash during CorrectDelayedTyposInExpr,
  // or any other use of TreeTransform that doesn't implement TransformDecl
  // separately.  Also, this should only error on 'output', not that 'x' is not
  // captured.
  // expected-error@+1 {{use of undeclared identifier 'output'}}
  all_of(output, [](T x) { return x; });
}

int main() {
  load_test<int>();
  return 0;
}
