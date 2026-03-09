// RUN: %check_clang_tidy %s modernize-use-return-value %t

struct Widget {
  int X;
};

// Positive: void function with single non-const ref output param.
void getInt(int &Out) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'getInt' has output parameter 'Out'
  Out = 42;
}

// Positive: struct output parameter.
void getWidget(Widget &Out) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'getWidget' has output parameter 'Out'
  Out.X = 10;
}

// Positive: mixed const and non-const ref params (one non-const).
void transform(const int &In, int &Out) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'transform' has output parameter 'Out'
  Out = In * 2;
}

// Positive: non-const ref + value params.
void compute(int A, int B, int &Result) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'compute' has output parameter 'Result'
  Result = A + B;
}

// Negative: returns non-void.
int getX(int &Out) {
  Out = 42;
  return 0;
}

// Negative: const ref parameter (input, not output).
void readOnly(const int &X) {
}

// Negative: multiple non-const ref params (ambiguous outparam).
void swap(int &A, int &B) {
  int T = A;
  A = B;
  B = T;
}

// Negative: no assignment to the parameter.
void noWrite(int &X) {
  int Y = X + 1;
}

// Negative: virtual method.
struct Base {
  virtual void vmethod(int &Out);
};

// Negative: abstract output type.
struct Abstract {
  virtual void foo() = 0;
};
void getAbstract(Abstract &Out) {
  // Not flagged -- Abstract cannot be returned by value.
}

// Negative: array type.
void getArray(int (&Out)[10]) {
  Out[0] = 1;
}

// Negative: unnamed parameter.
void unnamed(int &) {
}
