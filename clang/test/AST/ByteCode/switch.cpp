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

namespace gnurange {
  constexpr int l(int n) {
    return n + 1;
  }
  constexpr int h(int n) {
    return 2 * n + 1;
  }
  constexpr int f(int x) {
    const int n = 2;
    constexpr struct {
      char lo {'a'};
      char hi {'z'};
    } s;

    switch (x) {
      case l(n) ... h(n):
        return 1;
      case -1 ... 1:
        return 2;
      case 9 ... 14:
        return 3;
      case 15:
        return 4;
      case 16 ... 20:
        return 5;
      case s.lo ... s.hi:
        return 6;
      default:
        return -1;
    }
  }
  static_assert(f(0) == 2);
  static_assert(f(2) == -1);
  static_assert(f(3) == 1);
  static_assert(f(4) == 1);
  static_assert(f(5) == 1);
  static_assert(f(6) == -1);
  static_assert(f(14) == 3);
  static_assert(f(15) == 4);
  static_assert(f(16) == 5);
  static_assert(f(20) == 5);
  static_assert(f('d') == 6);

  template <int Lo, int Hi>
  constexpr bool g(int x) {
    switch (x) {
      case Lo ... Hi:
        break;
      default:
        return false;
    }
    return true;
  }
  static_assert(g<100, 200>(132));

  constexpr bool m(int x) {
    switch (x) {
      case 10 ... 1: // both-warning {{empty case range specified}}
        return true;
      default:
        return false;
    }
  }
  static_assert(m(3)); // both-error {{static assertion failed due to requirement 'm(3)'}}
  static_assert(!m(3));

  constexpr bool j(int x) { // both-note {{declared here}}
    switch (x) {
      case bad(x) ... 100: // both-error {{case value is not a constant expression}} \
                           // both-note {{cannot be used in a constant expression}}
        return true;
      default:
        break;
    }
    return false;
  }
  static_assert(j(1)); // both-error {{static assertion failed}}

  constexpr bool d(int x) { // both-note {{declared here}}
    switch (x) {
      case -100 ... bad(x): // both-error {{case value is not a constant expression}} \
                            // both-note {{cannot be used in a constant expression}}
        return true;
      default:
        break;
    }
    return false;
  }
  static_assert(d(1)); // both-error {{static assertion failed}}
  
  constexpr bool s(int x) { // both-note {{declared here}}
    switch (x) {
      case bad(x) - 100 ... bad(x) + 100: // both-error {{case value is not a constant expression}} \
                                          // both-note {{cannot be used in a constant expression}}
        return true;
      default:
        break;
    }
    return false;
  }
  static_assert(s(1)); // both-error {{static assertion failed}}
}
