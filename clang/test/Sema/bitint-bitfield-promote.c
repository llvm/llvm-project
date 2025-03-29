// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s

// GH87641 noticed that integer promotion of a bit-field of bit-precise integer
// type was promoting to int rather than the type of the bit-field.
struct S {
  unsigned _BitInt(7) x : 2;
  unsigned _BitInt(2) y : 2;
  unsigned _BitInt(72) z : 28;
  _BitInt(31) a : 12;
  _BitInt(33) b : 33;
};

// We don't have to worry about promotion cases where the bit-precise type is
// smaller than the width of the bit-field; that can't happen.
struct T {
  unsigned _BitInt(28) oh_no : 72; // expected-error {{width of bit-field 'oh_no' (72 bits) exceeds the width of its type (28 bits)}}
};

static_assert(
  _Generic(+(struct S){}.x,
    int : 0,
    unsigned _BitInt(7) : 1,
    unsigned _BitInt(2) : 2
  ) == 1);

static_assert(
  _Generic(+(struct S){}.y,
    int : 0,
    unsigned _BitInt(7) : 1,
    unsigned _BitInt(2) : 2
  ) == 2);

static_assert(
  _Generic(+(struct S){}.z,
    int : 0,
    unsigned _BitInt(72) : 1,
    unsigned _BitInt(28) : 2
  ) == 1);

static_assert(
  _Generic(+(struct S){}.a,
    int : 0,
    _BitInt(31) : 1,
    _BitInt(12) : 2,
    unsigned _BitInt(31) : 3
  ) == 1);

static_assert(
  _Generic(+(struct S){}.b,
    int : 0,
    long long : 1,
    _BitInt(33) : 2,
    unsigned _BitInt(33) : 3
  ) == 2);
