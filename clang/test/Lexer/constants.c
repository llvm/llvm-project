// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -trigraphs %s

int x = 000000080;  // expected-error {{invalid digit}}

int y = 0000\
00080;             // expected-error {{invalid digit}}



float X = 1.17549435e-38F;
float Y = 08.123456;

// PR2252
#if -0x8000000000000000  // should not warn.
#endif


int c[] = {
  'df',   // expected-warning {{multi-character character constant}}
  '\t',
  '\\
t',
  '??!',  // expected-warning {{trigraph converted to '|' character}}
  'abcd'  // expected-warning {{multi-character character constant}}
};

//  PR4499
int m0 = '0';
int m1 = '\\\''; // expected-warning {{multi-character character constant}}
int m2 = '\\\\'; // expected-warning {{multi-character character constant}}
int m3 = '\\\
';


#pragma clang diagnostic ignored "-Wmultichar"

int d = 'df'; // no warning.
int e = 'abcd';  // still warn: expected-warning {{multi-character character constant}}

#pragma clang diagnostic ignored "-Wfour-char-constants"

int f = 'abcd';  // ignored.

// rdar://problem/6974641
float t0[] = {
  1.9e20f,
  1.9e-20f,
  1.9e50f,   // expected-warning {{too large}}
  1.9e-50f,  // expected-warning {{too small}}
  -1.9e20f,
  -1.9e-20f,
  -1.9e50f,  // expected-warning {{too large}}
  -1.9e-50f  // expected-warning {{too small}}
};
double t1[] = {
  1.9e50,
  1.9e-50,
  1.9e500,   // expected-warning {{too large}}
  1.9e-500,  // expected-warning {{too small}}
  -1.9e50,
  -1.9e-50,
  -1.9e500,  // expected-warning {{too large}}
  -1.9e-500  // expected-warning {{too small}}
};

// PR7888
double g = 1e100000000; // expected-warning {{too large}}

char h = '\u1234'; // expected-error {{character too large for enclosing character literal type}}
