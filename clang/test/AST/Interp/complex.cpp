// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected -Wno-unused-value %s
// RUN: %clang_cc1 -verify=both,ref -Wno-unused-value %s

constexpr _Complex double z1 = {1.0, 2.0};
static_assert(__real(z1) == 1.0, "");
static_assert(__imag(z1) == 2.0, "");

static_assert(&__imag z1 == &__real z1 + 1, "");
static_assert((*(&__imag z1)) == __imag z1, "");
static_assert((*(&__real z1)) == __real z1, "");


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

static_assert(__real(4.0) == 4.0, "");
static_assert(__real(12u) == 12u, "");
static_assert(__imag(4.0) == 0.0, "");
static_assert(__imag(13) == 0, "");


constexpr _Complex long L1 = D;
static_assert(__real(L1) == 1.0, "");
static_assert(__imag(L1) == 3.0, "");

constexpr _Complex short I4 = L1;
static_assert(__real(I4) == 1, "");
static_assert(__imag(I4) == 3, "");

constexpr _Complex float D3 = D;
static_assert(__real(D3) == 1.0, "");
static_assert(__imag(D3) == 3.0, "");


constexpr _Complex int a = 2i;
static_assert(__real(a) == 0, "");
static_assert(__imag(a) == 2, "");

constexpr _Complex double b = 4.0i;
static_assert(__real(b) == 0, "");
static_assert(__imag(b) == 4, "");

constexpr int ignored() {
  I2;
  (int)I2;
  (float)I2;
  D1;
  (int)D1;
  (double)D1;
  (_Complex float)I2;
  (bool)D1;
  (bool)I2;
  return 0;
}
static_assert(ignored() == 0, "");
static_assert((int)I1 == 1, "");
static_assert((float)D == 1.0f, "");

static_assert(__real((_Complex unsigned)5) == 5);
static_assert(__imag((_Complex unsigned)5) == 0);

/// Standalone complex expressions.
static_assert(__real((_Complex float){1.0, 3.0}) == 1.0, "");


constexpr _Complex double D2 = {12};
static_assert(__real(D2) == 12, "");
static_assert(__imag(D2) == 0, "");

constexpr _Complex int I3 = {15};
static_assert(__real(I3) == 15, "");
static_assert(__imag(I3) == 0, "");

constexpr _Complex _BitInt(8) A = {4};
static_assert(__real(A) == 4, "");
static_assert(__imag(A) == 0, "");


constexpr _Complex double Doubles[4] = {{1.0, 2.0}};
static_assert(__real(Doubles[0]) == 1.0, "");
static_assert(__imag(Doubles[0]) == 2.0, "");
static_assert(__real(Doubles[1]) == 0.0, "");
static_assert(__imag(Doubles[1]) == 0.0, "");
static_assert(__real(Doubles[2]) == 0.0, "");
static_assert(__imag(Doubles[2]) == 0.0, "");
static_assert(__real(Doubles[3]) == 0.0, "");
static_assert(__imag(Doubles[3]) == 0.0, "");

void func(void) {
  __complex__ int arr;
  _Complex int result;
  int ii = 0;
  int bb = 0;
  /// The following line will call into the constant interpreter.
  result = arr * ii;
}

constexpr _Complex float getComplexFloat() {
  return {1,2};
}
static_assert(__real(getComplexFloat()) == 1, "");
static_assert(__imag(getComplexFloat()) == 2, "");

namespace CastToBool {
  constexpr _Complex int F = {0, 1};
  static_assert(F, "");
  constexpr _Complex int F2 = {1, 0};
  static_assert(F2, "");
  constexpr _Complex int F3 = {0, 0};
  static_assert(!F3, "");

  constexpr _Complex unsigned char F4 = {0, 1};
  static_assert(F4, "");
  constexpr _Complex unsigned char F5 = {1, 0};
  static_assert(F5, "");
  constexpr _Complex unsigned char F6 = {0, 0};
  static_assert(!F6, "");

  constexpr _Complex float F7 = {0, 1};
  static_assert(F7, "");
  constexpr _Complex float F8 = {1, 0};
  static_assert(F8, "");
  constexpr _Complex double F9 = {0, 0};
  static_assert(!F9, "");
}

namespace BinOps {
namespace Add {
  constexpr _Complex float A = { 13.0, 2.0 };
  constexpr _Complex float B = { 2.0, 1.0  };
  constexpr _Complex float C = A + B;
  static_assert(__real(C) == 15.0, "");
  static_assert(__imag(C) == 3.0, "");

  constexpr _Complex float D = B + A;
  static_assert(__real(D) == 15.0, "");
  static_assert(__imag(D) == 3.0, "");

  constexpr _Complex unsigned int I1 = { 5,  10 };
  constexpr _Complex unsigned int I2 = { 40, 2  };
  constexpr _Complex unsigned int I3 = I1 + I2;
  static_assert(__real(I3) == 45, "");
  static_assert(__imag(I3) == 12, "");

  static_assert(__real(A + 2.0) == 15, "");
  static_assert(__imag(A + 2.0) == 2, "");
  static_assert(__real(2.0 + A) == 15, "");
  static_assert(__imag(2.0 + A) == 2, "");

  static_assert(__real(D + 1) == 16, "");
  static_assert(__real(D + 1.0) == 16, "");
  constexpr _Complex double D2 = D + 3.0;
  static_assert(__real(D2) == 18.0, "");
  static_assert(__imag(D2) == 3.0, "");
  constexpr _Complex double D3 = 3.0 + D;
  static_assert(__real(D3) == 18.0, "");
  static_assert(__imag(D3) == 3.0, "");
}

namespace Sub {
  constexpr _Complex float A = { 13.0, 2.0 };
  constexpr _Complex float B = { 2.0, 1.0  };
  constexpr _Complex float C = A - B;
  static_assert(__real(C) == 11.0, "");
  static_assert(__imag(C) == 1.0, "");
  static_assert(__real(A - 2.0) == 11, "");
  static_assert(__real(2.0 - A) == -11, "");

  constexpr _Complex float D = B - A;
  static_assert(__real(D) == -11.0, "");
  static_assert(__imag(D) == -1.0, "");

  constexpr _Complex unsigned int I1 = { 5,  10 };
  constexpr _Complex unsigned int I2 = { 40, 2  };
  constexpr _Complex unsigned int I3 = I1 - I2;
  static_assert(__real(I3) == -35, "");
  static_assert(__imag(I3) == 8, "");

  using Bobble = _Complex float;
  constexpr _Complex float A_ = { 13.0, 2.0 };
  constexpr Bobble B_ = { 2.0, 1.0  };
  constexpr _Complex float D_ = A_ - B_;
  static_assert(__real(D_) == 11.0, "");
  static_assert(__imag(D_) == 1.0, "");

  static_assert(__real(D - 1) == -12, "");
  static_assert(__real(D - 1.0) == -12, "");
  constexpr _Complex double D2 = D - 3.0;
  static_assert(__real(D2) == -14.0, "");
  static_assert(__imag(D2) == -1.0, "");
  constexpr _Complex double D3 = 3.0 - D;
  static_assert(__real(D3) == 14.0, "");
  static_assert(__imag(D3) == 1.0, "");
}

}

namespace ZeroInit {
  typedef _Complex float fcomplex;
  typedef _Complex unsigned icomplex;

  constexpr fcomplex test7 = fcomplex();
  static_assert(__real(test7) == 0.0f, "");
  static_assert(__imag(test7) == 0.0f, "");

  constexpr icomplex test8 = icomplex();
  static_assert(__real(test8) == 0, "");
  static_assert(__imag(test8) == 0, "");

  constexpr int ignored = (fcomplex(), 0);
}

namespace DeclRefCopy {
  constexpr _Complex int ComplexInt = 42 + 24i;

  constexpr _Complex int B = ComplexInt;
  constexpr _Complex int ArrayOfComplexInt[4] = {ComplexInt, ComplexInt, ComplexInt, ComplexInt};
  static_assert(__real(ArrayOfComplexInt[0]) == 42, "");
  static_assert(__imag(ArrayOfComplexInt[0]) == 24, "");
  static_assert(__real(ArrayOfComplexInt[3]) == 42, "");
  static_assert(__imag(ArrayOfComplexInt[3]) == 24, "");

  constexpr int localComplexArray() {
    _Complex int A = 42 + 24i;
    _Complex int ArrayOfComplexInt[4] = {A, A, A, A};
    return __real(ArrayOfComplexInt[0]) + __imag(ArrayOfComplexInt[3]);
  }
  static_assert(localComplexArray() == (24 + 42), "");
}

namespace Builtin {
  constexpr _Complex float A = __builtin_complex(10.0f, 20.0f);
  static_assert(__real(A) == 10, "");
  static_assert(__imag(A) == 20, "");

  constexpr _Complex double B = __builtin_complex(10.0, 20.0);
  static_assert(__real(B) == 10, "");
  static_assert(__imag(B) == 20, "");


  constexpr _Complex float C = __builtin_complex(10.0f, 20.0); // both-error {{arguments are of different types}}
}

namespace Cmp {
  static_assert((0.0 + 0.0j) == (0.0 + 0.0j));
  static_assert((0.0 + 0.0j) != (0.0 + 0.0j)); // both-error {{static assertion}} \
                                               // both-note {{evaluates to}}

  static_assert((0.0 + 0.0j) == 0.0);
  static_assert(0.0 == (0.0 + 0.0j));
  static_assert(0.0 == 0.0j);
  static_assert((0.0 + 1.0j) != 0.0);
  static_assert(1.0 != (0.0 + 0.0j));
  static_assert(0.0 != 1.0j);

  // Walk around the complex plane stepping between angular differences and
  // equality.
  static_assert((1.0 + 0.0j) == (0.0 + 0.0j)); // both-error {{static assertion}} \
                                               // both-note {{evaluates to}}
  static_assert((1.0 + 0.0j) == (1.0 + 0.0j));
  static_assert((1.0 + 1.0j) == (1.0 + 0.0j)); // both-error {{static assertion}} \
                                               // both-note {{evaluates to}}
  static_assert((1.0 + 1.0j) == (1.0 + 1.0j));
  static_assert((0.0 + 1.0j) == (1.0 + 1.0j)); // both-error {{static assertion}} \
                                               // both-note {{evaluates to}}
  static_assert((0.0 + 1.0j) == (0.0 + 1.0j));
  static_assert((-1.0 + 1.0j) == (0.0 + 1.0j)); // both-error {{static assertion}} \
                                                // both-note {{evaluates to}}
  static_assert((-1.0 + 1.0j) == (-1.0 + 1.0j));
  static_assert((-1.0 + 0.0j) == (-1.0 + 1.0j)); // both-error {{static assertion}} \
                                                 // both-note {{evaluates to}}
  static_assert((-1.0 + 0.0j) == (-1.0 + 0.0j));
  static_assert((-1.0 - 1.0j) == (-1.0 + 0.0j)); // both-error {{static assertion}} \
                                                 // both-note {{evaluates to}}
  static_assert((-1.0 - 1.0j) == (-1.0 - 1.0j));
  static_assert((0.0 - 1.0j) == (-1.0 - 1.0j)); // both-error {{static assertion}} \
                                                // both-note {{evaluates to}}
  static_assert((0.0 - 1.0j) == (0.0 - 1.0j));
  static_assert((1.0 - 1.0j) == (0.0 - 1.0j)); // both-error {{static assertion}} \
                                               // both-note {{evaluates to}}
  static_assert((1.0 - 1.0j) == (1.0 - 1.0j));

  /// Make sure these are rejected before reaching the constexpr interpreter.
  static_assert((0.0 + 0.0j) & (0.0 + 0.0j)); // both-error {{invalid operands to binary expression}}
  static_assert((0.0 + 0.0j) | (0.0 + 0.0j)); // both-error {{invalid operands to binary expression}}
  static_assert((0.0 + 0.0j) < (0.0 + 0.0j)); // both-error {{invalid operands to binary expression}}
  static_assert((0.0 + 0.0j) > (0.0 + 0.0j)); // both-error {{invalid operands to binary expression}}
  static_assert((0.0 + 0.0j) ^ (0.0 + 0.0j)); // both-error {{invalid operands to binary expression}}
}

/// From test/SemaCXX/constant-expression-cxx11.cpp
///
/// Some of the diagnostics we emit are different than the one of the
/// current interpreter.
///
/// FIXME: For the '&test3 + 1' test, we are _not_ creating an explicit pointer variable
/// anywhere and so the &test3+1 is the same as __imag(test3) for us.
namespace ComplexConstexpr {
  constexpr _Complex float test1 = {};
  constexpr _Complex float test2 = {1};
  constexpr _Complex double test3 = {1,2};
  constexpr _Complex int test4 = {4};
  constexpr _Complex int test5 = 4;
  constexpr _Complex int test6 = {5,6};
  typedef _Complex float fcomplex;
  constexpr fcomplex test7 = fcomplex();

  constexpr const double &t2r = __real test3;
  constexpr const double &t2i = __imag test3;
  static_assert(&t2r + 1 == &t2i, "");
  static_assert(t2r == 1.0, "");
  static_assert(t2i == 2.0, "");
  constexpr const double *t2p = &t2r;
  static_assert(t2p[-1] == 0.0, ""); // both-error {{constant expr}} \
                                     // both-note {{cannot refer to element -1 of array of 2 elements}}
  static_assert(t2p[0] == 1.0, "");
  static_assert(t2p[1] == 2.0, "");
  static_assert(t2p[2] == 0.0, ""); // both-error {{constant expr}} \
                                    // both-note {{one-past-the-end pointer}}
  static_assert(t2p[3] == 0.0, ""); // both-error {{constant expr}} \
                                    // both-note {{cannot refer to element 3 of array of 2 elements}}
  constexpr _Complex float *p = 0;
  constexpr float pr = __real *p; // both-error {{constant expr}} \
                                  // ref-note {{cannot access real component of null}} \
                                  // expected-note {{read of dereferenced null pointer}}
  constexpr float pi = __imag *p; // both-error {{constant expr}} \
                                  // ref-note {{cannot access imaginary component of null}} \
                                  // expected-note {{cannot perform pointer arithmetic on null pointer}}
  constexpr const _Complex double *q = &test3 + 1;
  constexpr double qr = __real *q; // ref-error {{constant expr}} \
                                   // ref-note {{cannot access real component of pointer past the end}}
  constexpr double qi = __imag *q; // both-error {{constant expr}} \
                                   // ref-note {{cannot access imaginary component of pointer past the end}} \
                                   // expected-note {{read of dereferenced one-past-the-end pointer}}

  static_assert(__real test6 == 5, "");
  static_assert(__imag test6 == 6, "");
  static_assert(&__imag test6 == &__real test6 + 1, "");
}
