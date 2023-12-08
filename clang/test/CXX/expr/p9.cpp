// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// floating-point overloads

__typeof__(0 + 0.0L) ld0;
long double &ldr = ld0;

__typeof__(0 + 0.0) d0;
double &dr = d0;

__typeof__(0 + 0.0f) f0;
float &fr = f0;

// integral promotions

signed char c0;
__typeof__(c0 + c0) c1;
int &cr = c1;

unsigned char uc0;
__typeof__(uc0 + uc0) uc1;
int &ucr = uc1;

short s0;
__typeof__(s0 + s0) s1;
int &sr = s1;

unsigned short us0;
__typeof__(us0 + us0) us1;
int &usr = us1;

// integral overloads

__typeof__(0 + 0UL) ul0;
unsigned long &ulr = ul0;

template<bool T> struct selector;
template<> struct selector<true> { typedef long type; };
template<> struct selector<false> {typedef unsigned long type; };
__typeof__(0U + 0L) ui_l0;
selector<(sizeof(long) > sizeof(unsigned int))>::type &ui_lr = ui_l0;

__typeof__(0 + 0L) l0;
long &lr = l0;

__typeof__(0 + 0U) u0;
unsigned &ur = u0;

__typeof__(0 + 0) i0;
int &ir = i0;
