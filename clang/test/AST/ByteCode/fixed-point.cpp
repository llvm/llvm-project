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

namespace FloatToFixedPointCast {
  constexpr _Fract sf = 1.0; // both-error {{must be initialized by a constant expression}} \
                             // both-note {{outside the range of representable values of type 'const _Fract'}}

  constexpr _Fract sf2 = 0.5;
  static_assert(sf2 == 0.5);
  constexpr float sf2f = sf2;
  static_assert(sf2f == 0.5);
}
