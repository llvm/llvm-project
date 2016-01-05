// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

typedef unsigned char uchar;
typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));

typedef char char2 __attribute__((ext_vector_type(2)));
typedef char char3 __attribute__((ext_vector_type(3)));

typedef int int2 __attribute__((ext_vector_type(2)));

typedef float float2 __attribute__((ext_vector_type(2)));

// ** Positive tests **

// all scalars, but widths do not match.
int ptest01(char C, char X, int Y)
{
  return C ? X : Y;
}

char ptest02(int C, char X, char Y)
{
  return C ? X : Y;
}

// scalar condition and mixed-width vectors and scalars
int2 ptest03(char C, char X, int2 Y)
{
  return C ? X : Y;
}

// uniform vectors
char2 ptest04(char2 X, char2 Y, char2 C)
{
  return C ? X : Y;
}

// vector condition and mixed scalar operands
int2 ptest05(int2 C, int X, char Y)
{
  return C ? X : Y;
}

// vector condition and matching scalar operands
float2 ptest06(int2 C, float X, float Y)
{
  return C ? X : Y;
}

// vector condition and mixed scalar operands
float2 ptest07(int2 C, int X, float Y)
{
  return C ? X : Y;
}

// vector condition and mixed scalar and vector operands
float2 ptest08(int2 C, int X, float2 Y)
{
  return C ? X : Y;
}

// Actual comparison expression
float2 ptest09(float2 A, float2 B, float2 C, float2 D)
{
  return A < B ? C : D;
}

// ** Negative tests **

int2 ntest01(char2 C, int X, int Y)
{
  return C ? X : Y; // expected-error {{vector condition type 'char2' (vector of 2 'char' values) and result type (vector of 2 'int' values) do not have elements of the same size}}
}

int2 ntest02(char2 C, int2 X, int2 Y)
{
  return C ? X : Y; // expected-error {{vector condition type 'char2' (vector of 2 'char' values) and result type 'int2' (vector of 2 'int' values) do not have elements of the same size}}
}

uchar2 ntest03(int2 C, uchar X, uchar Y)
{
  return C ? X : Y; // expected-error {{vector condition type 'int2' (vector of 2 'int' values) and result type (vector of 2 'unsigned char' values) do not have elements of the same size}}
}

float2 ntest04(int2 C, int2 X, float2 Y)
{
  return C ? X : Y; // expected-error {{implicit conversions between vector types ('int2' (vector of 2 'int' values) and 'float2' (vector of 2 'float' values)) are not permitted}}
}

float2 ntest05(int2 C, int2 X, float Y)
{
  return C ? X : Y; // expected-error {{cannot convert between vector values of different size ('int2' (vector of 2 'int' values) and 'float')}}
}

char2 ntest06(int2 C, char2 X, char2 Y)
{
  return C ? X : Y; // expected-error {{vector condition type 'int2' (vector of 2 'int' values) and result type 'char2' (vector of 2 'char' values) do not have elements of the same size}}
}

float ntest07(float C, float X, float Y)
{
  return C ? X : Y; // expected-error {{used type 'float' where floating point type is not allowed}}
}

float2 ntest08(float2 C, float2 X, float2 Y)
{
  return C ? X : Y; // expected-error {{used type 'float2' (vector of 2 'float' values) where floating point type is not allowed}}
}

// Trying to create a int2 vector out of pointers.
int2 ntest09(int2 C, global int *X, global int *Y)
{
  return C ? X : Y; // expected-error {{used type '__global int *' where integer or floating point type is required}}
}

char3 ntest10(char C, char3 X, char2 Y)
{
  return C ? X : Y; // expected-error {{implicit conversions between vector types ('char3' (vector of 3 'char' values) and 'char2' (vector of 2 'char' values)) are not permitted}}
}

char3 ntest11(char2 C, char3 X, char Y)
{
  return C ? X : Y; // expected-error {{vector condition type 'char2' (vector of 2 'char' values) and result type 'char3' (vector of 3 'char' values) do not have the same number of elements}}
}

int foo1(int);
int foo2(int);

unsigned int ntest12(int2 C)
{
  return (unsigned int)(C ? foo1 : foo2); // expected-error {{taking address of function is not allowed}} expected-error {{taking address of function is not allowed}}
}
