// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=expected,both -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=ref,both

static_assert((bool)1.0k);
static_assert(!((bool)0.0k));
static_assert((bool)0.0k); // both-error {{static assertion failed}}

static_assert(1.0k == 1.0k);
static_assert(1.0k == 1);
static_assert(1.0k != 1.0k); // both-error {{failed due to requirement '1.0k != 1.0k'}}
static_assert(1.0k != 1); // both-error {{failed due to requirement '1.0k != 1'}}
static_assert(-12.0k == -(-(-12.0k)));

/// Zero-init.
constexpr _Accum A{};
static_assert(A == 0.0k);
static_assert(A == 0);

namespace IntToFixedPointCast {
  constexpr _Accum B = 13;
  static_assert(B == 13.0k);
  static_assert(B == 13);

  constexpr _Fract sf = -1;
  static_assert(sf == -1.0k);
  static_assert(sf == -1);
}

namespace FixedPointToIntCasts {
  constexpr _Accum A = -13.0k;
  constexpr int I = A;
  static_assert(I == -13);
}

namespace FloatToFixedPointCast {
  constexpr _Fract sf = 1.0; // both-error {{must be initialized by a constant expression}} \
                             // both-note {{outside the range of representable values of type 'const _Fract'}}

  constexpr _Fract sf2 = 0.5;
  static_assert(sf2 == 0.5);
  constexpr float sf2f = sf2;
  static_assert(sf2f == 0.5);
}

namespace BinOps {
  constexpr _Accum A = 13;
  static_assert(A + 1 == 14.0k);
  static_assert(1 + A == 14.0k);
  static_assert((A + A) == 26);

  static_assert(A + 100000 == 14.0k); // both-error {{is not an integral constant expression}} \
                                      // both-note {{is outside the range of representable values}}

  static_assert((A - A) == 0);
  constexpr short _Accum mul_ovf1 = 255.0hk * 4.5hk; // both-error {{must be initialized by a constant expression}} \
                                                     // both-note {{value 123.5 is outside the range of representable values of type 'short _Accum'}}
  constexpr short _Accum div_ovf1 = 255.0hk / 0.5hk; // both-error {{must be initialized by a constant expression}} \
                                                     // both-note {{value -2.0 is outside the range of representable values of type 'short _Accum'}}

}

namespace FixedPointCasts {
  constexpr _Fract B = 0.3;
  constexpr _Accum A = B;
  constexpr _Fract C = A;
}

namespace Cmp {
  constexpr _Accum A = 13.0k;
  constexpr _Accum B = 14.0k;
  static_assert(B > A);
  static_assert(B >= A);
  static_assert(A < B);
  static_assert(A <= B);
}
