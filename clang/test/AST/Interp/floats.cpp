// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s


constexpr void assert(bool C) {
  if (C)
    return;
  // Invalid in constexpr.
  (void)(1 / 0); // expected-warning {{undefined}} \
                 // ref-warning {{undefined}}
}

constexpr int i = 2;
constexpr float f = 1.0f;
static_assert(f == 1.0f, "");

constexpr float f2 = 1u * f;
static_assert(f2 == 1.0f, "");

constexpr float f3 = 1.5;
constexpr int i3 = f3;
static_assert(i3 == 1, "");

constexpr bool b3 = f3;
static_assert(b3, "");


static_assert(1.0f + 3u == 4, "");
static_assert(4.0f / 1.0f == 4, "");
static_assert(10.0f * false == 0, "");

constexpr float floats[] = {1.0f, 2.0f, 3.0f, 4.0f};

constexpr float m = 5.0f / 0.0f; // ref-error {{must be initialized by a constant expression}} \
                                 // ref-note {{division by zero}} \
                                 // expected-error {{must be initialized by a constant expression}} \
                                 // expected-note {{division by zero}}

static_assert(~2.0f == 3, ""); // ref-error {{invalid argument type 'float' to unary expression}} \
                               // expected-error {{invalid argument type 'float' to unary expression}}

/// Initialized by a double.
constexpr float df = 0.0;
/// The other way around.
constexpr double fd = 0.0f;

static_assert(0.0f == -0.0f, "");

const int k = 3 * (1.0f / 3.0f);
static_assert(k == 1, "");

constexpr bool b = 1.0;
static_assert(b, "");

constexpr double db = true;
static_assert(db == 1.0, "");

constexpr float fa[] = {1.0f, 2.0, 1, false};
constexpr double da[] = {1.0f, 2.0, 1, false};

constexpr float fm = __FLT_MAX__;
constexpr int someInt = fm; // ref-error {{must be initialized by a constant expression}} \
                            // ref-note {{is outside the range of representable values}} \
                            // expected-error {{must be initialized by a constant expression}} \
                            // expected-note {{is outside the range of representable values}}

namespace compound {
  constexpr float f1() {
    float f = 0;
    f += 3.0;
    f -= 3.0f;

    f += 1;
    f /= 1;
    f /= 1.0;
    f *= f;

    f *= 2.0;
    return f;
  }
  static_assert(f1() == 2, "");

  constexpr float f2() {
    float f = __FLT_MAX__;
    f += 1.0;
    return f;
  }
  static_assert(f2() == __FLT_MAX__, "");
}

namespace unary {
  constexpr float a() {
    float f = 0.0;
    assert(++f == 1.0);
    assert(f == 1.0);
    ++f;
    f++;
    assert(f == 3.0);
    --f;
    f--;
    assert(f == 1.0);
    return 1.0;
  }
  static_assert(a() == 1.0, "");

  constexpr float b() {
    float f = __FLT_MAX__;
    f++;
    return f;
  }
  static_assert(b() == __FLT_MAX__, "");
}


namespace ZeroInit {
  template<typename FloatT>
  struct A {
    int a;
    FloatT f;
  };

  constexpr A<float> a{12};
  static_assert(a.f == 0.0f, "");

  constexpr A<double> b{12};
  static_assert(a.f == 0.0, "");
};
