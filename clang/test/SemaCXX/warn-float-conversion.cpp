// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-literal-conversion -Wfloat-conversion -DFLOAT_CONVERSION -DZERO -DBOOL -DCONSTANT_BOOL -DOVERFLOW
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-overflow-conversion -DOVERFLOW
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-zero-conversion -DZERO
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-bool-constant-conversion -DCONSTANT_BOOL
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-bool-conversion -DCONSTANT_BOOL -DBOOL

float ReturnFloat();

#ifdef FLOAT_CONVERSION
bool ReturnBool(float f) {
  return f;  //expected-warning{{conversion}}
}

char ReturnChar(float f) {
  return f;  //expected-warning{{conversion}}
}

int ReturnInt(float f) {
  return f;  //expected-warning{{conversion}}
}

long ReturnLong(float f) {
  return f;  //expected-warning{{conversion}}
}

void Convert(float f, double d, long double ld) {
  bool b;
  char c;
  int i;
  long l;

  b = f;  //expected-warning{{conversion}}
  b = d;  //expected-warning{{conversion}}
  b = ld;  //expected-warning{{conversion}}
  c = f;  //expected-warning{{conversion}}
  c = d;  //expected-warning{{conversion}}
  c = ld;  //expected-warning{{conversion}}
  i = f;  //expected-warning{{conversion}}
  i = d;  //expected-warning{{conversion}}
  i = ld;  //expected-warning{{conversion}}
  l = f;  //expected-warning{{conversion}}
  l = d;  //expected-warning{{conversion}}
  l = ld;  //expected-warning{{conversion}}
}

void Test() {
  int a1 = 10.0/2.0;  //expected-warning{{conversion}}
  int a2 = 1.0/2.0;  //expected-warning{{conversion}}
  bool a3 = ReturnFloat();  //expected-warning{{conversion}}
  int a4 = 1e30 + 1;  //expected-warning{{conversion}}
}

void TestConstantFloat() {
  // Don't warn on exact floating literals.
  int a1 = 5.0;
  int a2 = 1e3;

  int a3 = 5.5;  // caught by -Wliteral-conversion
  int a4 = 500.44;  // caught by -Wliteral-convserion

  int b1 = 5.0 / 1.0;  //expected-warning{{conversion}}
  int b2 = 5.0 / 2.0;  //expected-warning{{conversion}}

  const float five = 5.0;

  int b3 = five / 1.0;  //expected-warning{{conversion}}
  int b4 = five / 2.0;  //expected-warning{{conversion}}
}
#endif  // FLOAT_CONVERSION

#ifdef CONSTANT_BOOL
const float pi = 3.1415;

void TestConstantBool() {
  bool b1 = 0.99f; // expected-warning {{implicit conversion from 'float' to 'bool' changes value from 0.99 to true}}
  bool b2 = 0.99; // expected-warning {{implicit conversion from 'double' to 'bool' changes value from 0.99 to true}}
  bool b3 = 0.0f; // expected-warning {{implicit conversion from 'float' to 'bool' changes value from 0 to false}}
  bool b4 = 0.0; // expected-warning {{implicit conversion from 'double' to 'bool' changes value from 0 to false}}
  bool b5 = 1.0f; // expected-warning {{implicit conversion from 'float' to 'bool' changes value from 1 to true}}
  bool b6 = 1.0; // expected-warning {{implicit conversion from 'double' to 'bool' changes value from 1 to true}}
  bool b7 = pi; // expected-warning {{implicit conversion from 'const float' to 'bool' changes value from 3.1415 to true}}
  bool b8 = pi - pi; // expected-warning {{implicit conversion from 'float' to 'bool' changes value from 0 to false}}
}
#endif  // CONSTANT_BOOL

#ifdef BOOL
const float E = 2.718;

float GetFloat();
double GetDouble();

void TestBool() {
  bool b1 = GetFloat(); // expected-warning {{implicit conversion turns floating-point number into boolean: 'float' to 'bool'}}
  bool b2 = GetDouble(); // expected-warning {{implicit conversion turns floating-point number into boolean: 'double' to 'bool'}}
  bool b3 = 0.0 * GetDouble(); // expected-warning {{implicit conversion turns floating-point number into boolean: 'double' to 'bool'}}
  bool b4 = GetFloat() + GetDouble(); // expected-warning {{implicit conversion turns floating-point number into boolean: 'double' to 'bool'}}
  bool b5 = E + GetFloat(); // expected-warning {{implicit conversion turns floating-point number into boolean: 'float' to 'bool'}}
}

#endif  // BOOL

#ifdef ZERO
void TestZero() {
  const float half = .5;
  int a1 = half;  // expected-warning{{implicit conversion from 'const float' to 'int' changes non-zero value from 0.5 to 0}}
  int a2 = 1.0 / 2.0;  // expected-warning{{implicit conversion from 'double' to 'int' changes non-zero value from 0.5 to 0}}
  int a3 = 5;
}
#endif  // ZERO

#ifdef OVERFLOW
void TestOverflow() {
  char a = 500.0;  // caught by -Wliteral-conversion
  char b = -500.0;  // caught by -Wliteral-conversion

  const float LargeNumber = 1024;
  char c = LargeNumber;  // expected-warning{{implicit conversion of out of range value from 'const float' to 'char' changes value from 1024 to 127}}
  char d = 400.0 + 400.0;  // expected-warning{{implicit conversion of out of range value from 'double' to 'char' changes value from 800 to 127}}

  char e = 1.0 / 0.0;  // expected-warning{{implicit conversion of out of range value from 'double' to 'char' changes value from +Inf to 127}}
}
#endif  // OVERFLOW
