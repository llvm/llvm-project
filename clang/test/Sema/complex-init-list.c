// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic

// This file tests the clang extension which allows initializing the components
// of a complex number individually using an initialization list. Basically,
// if you have an explicit init list for a complex number that contains two
// initializers, this extension kicks in to turn it into component-wise
// initialization.
//
// This extension is useful because there isn't any way to accurately build
// a complex number at the moment besides setting the components with
// __real__ and __imag__, which is inconvenient and not usable for constants.
// (Of course, there are other extensions we could implement that would
// allow this, like some sort of __builtin_build_complex.)
//
// FIXME: It would be a good idea to have a warnings for implicit
// real->complex and complex->real conversions; as-is, it's way too easy
// to get implicit conversions when they are not intended.

// Basic testcase
_Complex float valid1 = { 1.0f, 2.0f }; // expected-warning {{specifying real and imaginary components is an extension}}


// Struct for nesting tests
struct teststruct { _Complex float x; };


// Random other valid stuff
_Complex int valid2 = { 1, 2 }; // expected-warning {{complex integer}} \
                               //  expected-warning {{specifying real and imaginary components is an extension}}
struct teststruct valid3 = { { 1.0f, 2.0f} }; // expected-warning {{specifying real and imaginary components is an extension}}
_Complex float valid4[2] = { {1.0f, 1.0f}, {1.0f, 1.0f} }; // expected-warning 2 {{specifying real and imaginary components is an extension}}
// FIXME: We need some sort of warning for valid5
_Complex float valid5 = {1.0f, 1.0fi}; // expected-warning {{imaginary constants}} \
                                       // expected-warning {{specifying real and imaginary components is an extension}}


// Random invalid stuff
struct teststruct invalid1 = { 1, 2 }; // expected-warning {{excess elements}}
_Complex float invalid2 = { 1, 2, 3 }; // expected-warning {{specifying real and imaginary components is an extension}} \
                                       // expected-warning {{excess elements in scalar initializer}}
_Complex float invalid3 = {}; // expected-warning {{use of an empty initializer is a C23 extension}}


// Check incomplete array sizing
_Complex float sizetest1[] = { {1.0f, 1.0f}, {1.0f, 1.0f} }; // expected-warning 2 {{specifying real and imaginary components is an extension}}
_Complex float sizecheck1[(sizeof(sizetest1) == sizeof(*sizetest1)*2) ? 1 : -1];
_Complex float sizetest2[] = { 1.0f, 1.0f, {1.0f, 1.0f} };  // expected-warning {{specifying real and imaginary components is an extension}}
_Complex float sizecheck2[(sizeof(sizetest2) == sizeof(*sizetest2)*3) ? 1 : -1];

// Constant-folding with init list.
_Complex float x = 2 + (_Complex float) { 1, 2 };  // expected-warning {{specifying real and imaginary components is an extension}}

// initialization list
_Complex double cd = {1.0, 2.0, 3.0}; // expected-warning {{specifying real and imaginary components is an extension}} \
                                     //  expected-warning {{excess elements in scalar initializer}}
_Complex float cf = {1.1f, 2.2f, 3.3f, 4.4f}; // expected-warning {{specifying real and imaginary components is an extension}} \
                                             //  expected-warning {{excess elements in scalar initializer}}
