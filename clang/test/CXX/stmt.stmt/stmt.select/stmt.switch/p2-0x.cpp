// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

struct Value {
  constexpr Value(int n) : n(n) {}
  constexpr operator short() const { return n; }
  int n;
};
enum E { E0, E1 };
struct Alt {
  constexpr operator E() const { return E0; }
};

constexpr short s = Alt();

void test(Value v) {
  switch (v) {
    case Alt():
    case E1:
    case Value(2):
    case 3:
      break;
  }
  switch (Alt a = Alt()) {
    case Alt():
    case E1:
    case Value(2):
    case 3:
      break;
  }
  switch (E0) {
    case Alt():
    case E1:
    // FIXME: These should produce a warning that 2 and 3 are not values of the
    // enumeration.
    case Value(2):
    case 3:
      break;
  }
}
