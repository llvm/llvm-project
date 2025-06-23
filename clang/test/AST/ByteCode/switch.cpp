// RUN: %clang_cc1 -std=c++17 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++17 -verify=ref,both %s

constexpr bool isEven(int a) {
  bool v = false;
  switch(a) {
  case 2: return true;
  case 4: return true;
  case 6: return true;

  case 8:
  case 10:
  case 12:
  case 14:
  case 16:
    return true;
  case 18:
    v = true;
    break;

  default:
  switch(a) {
  case 1:
    break;
  case 3:
    return false;
  default:
    break;
  }
  }

  return v;
}
static_assert(isEven(2), "");
static_assert(isEven(8), "");
static_assert(isEven(10), "");
static_assert(isEven(18), "");
static_assert(!isEven(1), "");
static_assert(!isEven(3), "");


constexpr int withInit() {
  switch(int a = 2; a) {
    case 1: return -1;
    case 2: return 2;
  }
  return -1;
}
static_assert(withInit() == 2, "");

constexpr int FT(int a) {
  int m = 0;
  switch(a) {
  case 4: m++;
  case 3: m++;
  case 2: m++;
  case 1: m++;
    return m;
  }

  return -1;
}
static_assert(FT(1) == 1, "");
static_assert(FT(4) == 4, "");
static_assert(FT(5) == -1, "");


constexpr int good() { return 1; }
constexpr int test(int val) {
  switch (val) {
  case good(): return 100;
  default: return -1;
  }
  return 0;
}
static_assert(test(1) == 100, "");

constexpr int bad(int val) { return val / 0; } // both-warning {{division by zero}}
constexpr int another_test(int val) { // both-note {{declared here}}
  switch (val) {
  case bad(val): return 100; // both-error {{case value is not a constant expression}} \
                             // both-note {{cannot be used in a constant expression}}
  default: return -1;
  }
  return 0;
}
static_assert(another_test(1) == 100, ""); // both-error {{static assertion failed}} \
                                           // both-note {{evaluates to}}
