// RUN: %check_clang_tidy %s bugprone-integer-division %t

// Functions expecting a floating-point parameter.
void floatArg(float x) {}
void doubleArg(double x) {}
void longDoubleArg(long double x) {}

// Functions expected to return a floating-point value.
float singleDiv() {
  int x = -5;
  int y = 2;
  return x/y;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of integer division used in
}

double wrongOrder(int x, int y) {
  return x/y/0.1;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of integer division used in
}

long double rightOrder(int x, int y) {
  return 0.1/x/y; // OK
}

// Typical mathematical functions.
float sin(float);
double acos(double);
long double tanh(long double);

namespace std {
  using ::sin;
}

template <typename T>
void intDivSin(T x) {
  sin(x);
}

int intFunc(int);

struct X {
  int n;
  void m() {
    sin(n / 3);
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: result of integer division used in
  }
};

void integerDivision() {
  char a = 2;
  short b = -5;
  int c = 9784;
  enum third { x, y, z=2 };
  third d = z;
  char e[] = {'a', 'b', 'c'};
  char f = *(e + 1 / a);
  bool g = 1;

  sin(1 + c / (2 + 2));
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of integer division used in
  sin(c / (1 + .5));
  sin((c + .5) / 3);

  sin(intFunc(3) / 5);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of integer division used in
  acos(2 / intFunc(7));
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of integer division used in

  floatArg(1 + 2 / 3);
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: result of integer division used in
  sin(1 + 2 / 3);
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of integer division used in
  intFunc(sin(1 + 2 / 3));
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: result of integer division used in

  floatArg(1 + intFunc(1 + 2 / 3));
  floatArg(1 + 3 * intFunc(a / b));

  1 << (2 / 3);
  1 << intFunc(2 / 3);

#define M_SIN sin(a / b);
  M_SIN
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: result of integer division used in

  intDivSin<float>(a / b);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: result of integer division used in
  intDivSin<double>(c / d);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: result of integer division used in
  intDivSin<long double>(f / g);
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: result of integer division used in

  floatArg(1 / 3);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of integer division used in
  doubleArg(a / b);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of integer division used in
  longDoubleArg(3 / d);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: result of integer division used in
  floatArg(a / b / 0.1);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of integer division used in
  doubleArg(1 / 3 / 0.1);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of integer division used in
  longDoubleArg(2 / 3 / 5);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: result of integer division used in

  std::sin(2 / 3);
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of integer division used in
  ::acos(7 / d);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of integer division used in
  tanh(f / g);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of integer division used in

  floatArg(0.1 / a / b);
  doubleArg(0.1 / 3 / 1);

  singleDiv();
  wrongOrder(a,b);
  rightOrder(a,b);

  sin(a / b);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of integer division used in
  acos(f / d);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of integer division used in
  tanh(c / g);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of integer division used in

  sin(3.0 / a);
  acos(b / 3.14);
  tanh(3.14 / f / g);
}
