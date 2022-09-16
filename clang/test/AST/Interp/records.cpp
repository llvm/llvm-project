// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

// ref-no-diagnostics
// expected-no-diagnostics

struct BoolPair {
  bool first;
  bool second;
};

struct Ints {
  int a = 20;
  int b = 30;
  bool c = true;
  // BoolPair bp = {true, false}; FIXME

  static const int five = 5;
  static constexpr int getFive() {
    return five;
  }

  constexpr int getTen() const {
    return 10;
  }
};

static_assert(Ints::getFive() == 5, "");

constexpr Ints ints;
static_assert(ints.a == 20, "");
static_assert(ints.b == 30, "");
static_assert(ints.c, "");
static_assert(ints.getTen() == 10, "");

constexpr Ints ints2{-20, -30, false};
static_assert(ints2.a == -20, "");
static_assert(ints2.b == -30, "");
static_assert(!ints2.c, "");

#if 0
constexpr Ints getInts() {
  return {64, 128, true};
}
constexpr Ints ints3 = getInts();
static_assert(ints3.a == 64, "");
static_assert(ints3.b == 128, "");
static_assert(ints3.c, "");
#endif

constexpr Ints ints4 = {
  .a = 40 * 50,
  .b = 0,
  .c = (ints.a > 0),

};
static_assert(ints4.a == (40 * 50), "");
static_assert(ints4.b == 0, "");
static_assert(ints4.c, "");


// FIXME: Implement initialization by DeclRefExpr.
//constexpr Ints ints4 = ints3;  TODO



struct Ints2 {
  int a = 10;
  int b;
};
// FIXME: Broken in the new constant interpreter.
//   Should be rejected, but without asan errors.
//constexpr Ints2 ints2;

class C {
  public:
    int a;
    int b;

  constexpr C() : a(100), b(200) {}
};

constexpr C c;
static_assert(c.a == 100, "");
static_assert(c.b == 200, "");

constexpr int getB() {
  C c;
  int &j = c.b;

  j = j * 2;

  return c.b;
}
static_assert(getB() == 400, "");

constexpr int getA(const C &c) {
  return c.a;
}
static_assert(getA(c) == 100, "");

constexpr const C* getPointer() {
  return &c;
}
static_assert(getPointer()->a == 100, "");
