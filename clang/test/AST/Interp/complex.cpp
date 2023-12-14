// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

// expected-no-diagnostics
// ref-no-diagnostics

constexpr _Complex double z1 = {1.0, 2.0};
static_assert(__real(z1) == 1.0, "");
static_assert(__imag(z1) == 2.0, "");

constexpr double setter() {
  _Complex float d = {1.0, 2.0};

  __imag(d) = 4.0;
  return __imag(d);
}
static_assert(setter() == 4, "");

constexpr _Complex double getter() {
  return {1.0, 3.0};
}
constexpr _Complex double D = getter();
static_assert(__real(D) == 1.0, "");
static_assert(__imag(D) == 3.0, "");


constexpr _Complex int I1 = {1, 2};
static_assert(__real(I1) == 1, "");
static_assert(__imag(I1) == 2, "");


constexpr _Complex double D1 = {};
static_assert(__real(D1) == 0, "");
static_assert(__imag(D1) == 0, "");

constexpr _Complex int I2 = {};
static_assert(__real(I2) == 0, "");
static_assert(__imag(I2) == 0, "");


#if 0
/// FIXME: This should work in the new interpreter.
constexpr _Complex double D2 = {12};
static_assert(__real(D2) == 12, "");
static_assert(__imag(D2) == 12, "");

constexpr _Complex int I3 = {15};
static_assert(__real(I3) == 15, "");
static_assert(__imag(I3) == 15, "");
#endif




/// FIXME: This should work in the new interpreter as well.
// constexpr _Complex _BitInt(8) A = 0;// = {4};
