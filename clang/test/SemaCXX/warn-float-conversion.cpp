// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-literal-conversion -Wfloat-conversion -DFLOAT_CONVERSION -DZERO -DBOOL -DCONSTANT_BOOL -DOVERFLOW
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-overflow-conversion -DOVERFLOW
// RUN: %clang_cc1 -verify -fsyntax-only -triple x86_64-pc-linux-gnu %s -Wno-conversion -Wfloat-zero-conversion -DZERO

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
