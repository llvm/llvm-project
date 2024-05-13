// RUN: %clang_cc1 -fsyntax-only -verify %s -triple i686-linux-gnu -target-feature -x87
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple i686-linux-gnu -DNOERROR

#ifdef NOERROR
// expected-no-diagnostics
#endif

typedef long double long_double;

// Declaration is fine, unless it is called or defined.
double decl(long_double x, long_double y);

template <typename T>
T decl_ld_del(T);

// No code is generated for deleted functions
long_double decl_ld_del(long_double) = delete;
double decl_ld_del(double) = delete;
float decl_ld_del(float) = delete;

#ifndef NOERROR
// expected-error@+4{{'def' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
// expected-note@+3{{'def' defined here}}
// expected-note@+2{{'x' defined here}}
#endif
int def(long_double x) {
#ifndef NOERROR
// expected-error@+2{{'x' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  return (int)x;
}

#ifndef NOERROR
// expected-note@+3{{'ld_args' defined here}}
// expected-note@+2{{'ld_args' defined here}}
#endif
int ld_args(long_double x, long_double y);

int call1(float x, float y) {
#ifndef NOERROR
  // expected-error@+2 2{{'ld_args' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  return ld_args(x, y);
}

#ifndef NOERROR
// expected-note@+2{{'ld_ret' defined here}}
#endif
long_double ld_ret(double x, double y);

int call2(float x, float y) {
#ifndef NOERROR
  // expected-error@+2{{'ld_ret' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  return (int)ld_ret(x, y);
}

int binop(double x, double y) {
#ifndef NOERROR
  // expected-error@+2 2{{expression requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  double z = (long_double)x * (long_double)y;
  return (int)z;
}

void assign1(long_double *ret, double x) {
#ifndef NOERROR
  // expected-error@+2{{expression requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  *ret = x;
}

struct st_long_double1 {
#ifndef NOERROR
  // expected-note@+2{{'ld' defined here}}
#endif
  long_double ld;
};

struct st_long_double2 {
#ifndef NOERROR
  // expected-note@+2{{'ld' defined here}}
#endif
  long_double ld;
};

struct st_long_double3 {
#ifndef NOERROR
  // expected-note@+2{{'ld' defined here}}
#endif
  long_double ld;
};

void assign2() {
  struct st_long_double1 st;
#ifndef NOERROR
  // expected-error@+3{{expression requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
  // expected-error@+2{{'ld' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  st.ld = 0.42;
}

void assign3() {
  struct st_long_double2 st;
#ifndef NOERROR
  // expected-error@+3{{expression requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
  // expected-error@+2{{'ld' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  st.ld = 42;
}

void assign4(double d) {
  struct st_long_double3 st;
#ifndef NOERROR
  // expected-error@+3{{expression requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
  // expected-error@+2{{'ld' requires  'long_double' (aka 'long double') type support, but target 'i686-unknown-linux-gnu' does not support it}}
#endif
  st.ld = d;
}

void assign5() {
  // unused variable declaration is fine
  long_double ld = 0.42;
}

double d_ret1(float x) {
  return 0.0;
}

double d_ret2(float x);

int d_ret3(float x) {
  return (int)d_ret2(x);
}

float f_ret1(float x) {
  return 0.0f;
}

float f_ret2(float x);

int f_ret3(float x) {
  return (int)f_ret2(x);
}
