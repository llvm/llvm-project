// RUN: %clang_cc1 -std=c2x -verify %s

_Static_assert(1'2'3 == 12'3, "");
_Static_assert(1'000'000 == 0xf'4240, "");
_Static_assert(0'004'000'000 == 0x10'0000, "");
_Static_assert(0b0101'0100 == 0x54, "");

int a0 = 123'; //'; // expected-error {{expected ';'}}
int b0 = 0'xff; // expected-error {{digit separator cannot appear at end of digit sequence}} expected-error {{suffix 'xff' on integer}}
int c0 = 0x'ff; // expected-error {{suffix 'x'ff' on integer}}
int d0 = 0'1234; // ok, octal
int e0 = 0'b1010; // expected-error {{digit 'b' in octal constant}}
int f0 = 0b'1010; // expected-error {{invalid digit 'b' in octal}}
int h0 = 0x1e+1; // expected-error {{invalid suffix '+1' on integer constant}}
int i0 = 0x1'e+1; // ok, 'e+' is not recognized after a digit separator

float a1 = 1'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
float b1 = 1'0e1;
float c1 = 1.'0e1; // expected-error {{digit separator cannot appear at start of digit sequence}}
float d1 = 1.0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
float e1 = 1e'1; // expected-error {{digit separator cannot appear at start of digit sequence}}
float g1 = 0.'0; // expected-error {{digit separator cannot appear at start of digit sequence}}
float h1 = .'0; // '; // expected-error {{expected expression}}, lexed as . followed by character literal
float i1 = 0x.'0p0; // expected-error {{digit separator cannot appear at start of digit sequence}}
float j1 = 0x'0.0p0; // expected-error {{invalid suffix 'x'0.0p0'}}
float k1 = 0x0'.0p0; // '; // expected-error {{expected ';'}}
float l1 = 0x0.'0p0; // expected-error {{digit separator cannot appear at start of digit sequence}}
float m1 = 0x0.0'p0; // expected-error {{digit separator cannot appear at end of digit sequence}}
float n1 = 0x0.0p'0; // expected-error {{digit separator cannot appear at start of digit sequence}}
float p1 = 0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
float q1 = 0'0e1;
float r1 = 0.'0e1; // expected-error {{digit separator cannot appear at start of digit sequence}}
float s1 = 0.0'e1; // expected-error {{digit separator cannot appear at end of digit sequence}}
float t1 = 0.0e'1; // expected-error {{digit separator cannot appear at start of digit sequence}}
float u1 = 0x.'p1f; // expected-error {{hexadecimal floating constant requires a significand}}
float v1 = 0e'f; // expected-error {{exponent has no digits}}
float w1 = 0x0p'f; // expected-error {{exponent has no digits}}
float x1 = 0'e+1; // expected-error {{digit separator cannot appear at end of digit sequence}}
float y1 = 0x0'p+1; // expected-error {{digit separator cannot appear at end of digit sequence}}

#line 123'456
_Static_assert(__LINE__ == 123456, "");

// UCNs can appear before digit separators but not after.
int a2 = 0\u1234'5; // expected-error {{invalid suffix '\u1234'5' on integer constant}}
int b2 = 0'\u12345; // '; // expected-error {{expected ';'}}

// extended characters can appear before digit separators but not after.
int a3 = 0ሴ'5; // expected-error {{invalid suffix 'ሴ'5' on integer constant}}
int b3 = 0'ሴ5; // '; // expected-error {{expected ';'}}

