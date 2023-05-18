// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
struct {unsigned x : 2;} x;
__typeof__((x.x+=1)+1) y;
__typeof__(x.x<<1) y;
int y;


struct { int x : 8; } x1;
long long y1;
__typeof__(((long long)x1.x + 1)) y1;


// Check for extensions: variously sized unsigned bit-fields fitting
// into a signed int promote to signed int.
enum E { ec1, ec2, ec3 };
struct S {
  enum E          e : 2;
  unsigned short us : 4;
  unsigned long long ul1 : 8;
  unsigned long long ul2 : 50;
} s;

__typeof(s.e + s.e) x_e;
int x_e;

__typeof(s.us + s.us) x_us;
int x_us;

__typeof(s.ul1 + s.ul1) x_ul1;
int x_ul1;

__typeof(s.ul2 + s.ul2) x_ul2;
unsigned long long x_ul2;

