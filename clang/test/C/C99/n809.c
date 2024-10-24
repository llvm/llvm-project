// RUN: %clang_cc1 -verify -std=c99 %s

/* WG14 N620, N638, N657, N694, N809: Partial
 * Complex and imaginary support in <complex.h>
 *
 * NB: Clang supports _Complex but not _Imaginary. In C99, _Complex support is
 * required outside of freestanding, but _Imaginary support is fully optional.
 * In C11, both are made fully optional.
 *
 * NB: _Complex support requires an underlying support library such as
 * compiler-rt to provide functions like __divsc3. Compiler-rt is not supported
 * on Windows.
 *
 * Because the functionality is so intertwined between the various papers,
 * we're testing all of the functionality in one file.
 */

// Demonstrate that we support spelling complex floating-point objects.
float _Complex f1;
_Complex float f2;

double _Complex d1;
_Complex double d2;

long double _Complex ld1;
_Complex long double ld2;

// Show that we don't support spelling imaginary types.
float _Imaginary fi1; // expected-error {{imaginary types are not supported}}
_Imaginary float fi2; // expected-error {{imaginary types are not supported}}

double _Imaginary di1; // expected-error {{imaginary types are not supported}}
_Imaginary double di2; // expected-error {{imaginary types are not supported}}

long double _Imaginary ldi1; // expected-error {{imaginary types are not supported}}
_Imaginary long double ldi2; // expected-error {{imaginary types are not supported}}

// Each complex type has the same representation and alignment as an array
// containing two elements of the corresponding real type. Note, it is not
// mandatory that the alignment of a structure containing an array of two
// elements has the same alignment as an array of two elements outside of a
// structure, but this is a property Clang supports.
_Static_assert(sizeof(float _Complex) == sizeof(struct { float mem[2]; }), "");
_Static_assert(_Alignof(float _Complex) == _Alignof(struct { float mem[2]; }), "");

_Static_assert(sizeof(double _Complex) == sizeof(struct { double mem[2]; }), "");
_Static_assert(_Alignof(double _Complex) == _Alignof(struct { double mem[2]; }), "");

_Static_assert(sizeof(long double _Complex) == sizeof(struct { long double mem[2]; }), "");
_Static_assert(_Alignof(long double _Complex) == _Alignof(struct { long double mem[2]; }), "");

// The first element corresponds to the real part and the second element
// corresponds to the imaginary part.
_Static_assert(__real((float _Complex){ 1.0f, 2.0f }) == 1.0f, "");
_Static_assert(__imag((float _Complex){ 1.0f, 2.0f }) == 2.0f, "");

_Static_assert(__real((double _Complex){ 1.0, 2.0 }) == 1.0, "");
_Static_assert(__imag((double _Complex){ 1.0, 2.0 }) == 2.0, "");

_Static_assert(__real((long double _Complex){ 1.0L, 2.0L }) == 1.0L, "");
_Static_assert(__imag((long double _Complex){ 1.0L, 2.0L }) == 2.0L, "");

// When a real value is converted to a complex value, the real part follows the
// usual conversion rules and the imaginary part should be zero.
_Static_assert(__real((float _Complex)1.0f) == 1.0f, "");
_Static_assert(__imag((float _Complex)1.0f) == 0.0f, "");

_Static_assert(__real((double _Complex)1.0f) == 1.0, "");
_Static_assert(__imag((double _Complex)1.0f) == 0.0, "");

_Static_assert(__real((long double _Complex)1.0f) == 1.0L, "");
_Static_assert(__imag((long double _Complex)1.0f) == 0.0L, "");

// When a complex value is converted to a real value, the real part follows the
// usual conversion rules and the imaginary part is discarded.
_Static_assert((float)(float _Complex){ 1.0f, 2.0f } == 1.0f, "");
_Static_assert((double)(float _Complex){ 1.0f, 2.0f } == 1.0, "");
_Static_assert((long double)(float _Complex){ 1.0f, 2.0f } == 1.0L, "");

// Complex values are only equal if both the real and imaginary parts are equal.
_Static_assert((float _Complex){ 1.0f, 2.0f } == (float _Complex){ 1.0f, 2.0f }, "");
_Static_assert((double _Complex){ 1.0, 2.0 } == (double _Complex){ 1.0, 2.0 }, "");
_Static_assert((long double _Complex){ 1.0L, 2.0L } == (long double _Complex){ 1.0L, 2.0L }, "");

_Static_assert((float _Complex){ 1.0f, 2.0f } != (float _Complex){ 2.0f, 0.0f }, "");
_Static_assert((double _Complex){ 1.0, 2.0 } != (double _Complex){ 2.0, 0.0 }, "");
_Static_assert((long double _Complex){ 1.0L, 2.0L } != (long double _Complex){ 2.0L, 0.0L }, "");

// You cannot use relational operator on complex values.
int i1 = (float _Complex){ 1.0f, 2.0f } < 10;        // expected-error {{invalid operands to binary expression}}
int i2 = (double _Complex){ 1.0f, 2.0f } > 10;       // expected-error {{invalid operands to binary expression}}
int i3 = (long double _Complex){ 1.0f, 2.0f } <= 10; // expected-error {{invalid operands to binary expression}}
int i4 = (float _Complex){ 1.0f, 2.0f } >= 10;       // expected-error {{invalid operands to binary expression}}

// As a type specifier, _Complex cannot appear alone; however, we support it as
// an extension by assuming _Complex double.
_Complex c = 1.0f; // expected-warning {{plain '_Complex' requires a type specifier; assuming '_Complex double'}}
// Because we don't support imaginary types, we don't extend the extension to
// that type specifier.
// FIXME: the warning diagnostic here is incorrect and should not be emitted.
_Imaginary i = 1.0f; // expected-warning {{plain '_Complex' requires a type specifier; assuming '_Complex double'}} \
                        expected-error {{imaginary types are not supported}}

void func(void) {
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wpedantic"
  // Increment and decrement operators have a constraint that their operand be
  // a real type; Clang supports this as an extension on complex types as well.
  _Complex float cf = 0.0f;

  cf++; // expected-warning {{'++' on an object of complex type is a C2y extension}}
  ++cf; // expected-warning {{'++' on an object of complex type is a C2y extension}}

  cf--; // expected-warning {{'--' on an object of complex type is a C2y extension}}
  --cf; // expected-warning {{'--' on an object of complex type is a C2y extension}}

  // However, unary + and - are fine, as is += 1.
  (void)-cf;
  (void)+cf;
  cf += 1;
#pragma clang diagnostic pop
}
