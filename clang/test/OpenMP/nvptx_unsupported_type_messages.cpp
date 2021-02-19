// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only

struct T {
  char a;
#ifndef _ARCH_PPC
  // expected-note@+1 {{'f' defined here}}
  __float128 f;
#else
  // expected-note@+1 {{'f' defined here}}
  long double f;
#endif
  char c;
  T() : a(12), f(15) {}
#ifndef _ARCH_PPC
// expected-error@+5 {{'f' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#else
// expected-error@+3 {{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#endif
  T &operator+(T &b) {
    f += b.a;
    return *this;
  }
};

struct T1 {
  char a;
  __int128 f;
  __int128 f1;
  char c;
  T1() : a(12), f(15) {}
  T1 &operator/(T1 &b) {
    f /= b.a;
    return *this;
  }
};

#ifndef _ARCH_PPC
// expected-error@+2 {{'boo' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 2{{'boo' defined here}}
void boo(__float128 A) { return; }
#else
// expected-error@+2 {{'boo' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 2{{'boo' defined here}}
void boo(long double A) { return; }
#endif
#pragma omp declare target
T a = T();
T f = a;
void foo(T a = T()) {
  a = a + f; // expected-note {{called by 'foo'}}
#ifndef _ARCH_PPC
// expected-error@+5 {{'boo' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#else
// expected-error@+3 {{'boo' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#endif
// expected-note@+1 {{called by 'foo'}}
  boo(0);
  return;
}
T bar() {
  return T();
}

void baz() {
  T t = bar();
}
T1 a1 = T1();
T1 f1 = a1;
void foo1(T1 a = T1()) {
  a = a / f1;
  return;
}
T1 bar1() {
  return T1();
}
void baz1() {
  T1 t = bar1();
}

// TODO: We should not emit an error for dead functions we do not emit.
inline void dead_inline_declare_target() {
// expected-note@+1 {{'b' defined here}}
  long double *a, b = 0;
// expected-error@+1 {{'b' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  a = &b;
}
// TODO: We should not emit an error for dead functions we do not emit.
static void dead_static_declare_target() {
// expected-note@+1 {{'b' defined here}}
  long double *a, b = 0;
// expected-error@+1 {{'b' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  a = &b;
}
template<bool>
void dead_template_declare_target() {
  long double *a, b = 0;
  a = &b;
}

// expected-note@+2 {{'ld_return1a' defined here}}
// expected-error@+1 {{'ld_return1a' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
long double ld_return1a() { return 0; }
// expected-note@+2 {{'ld_arg1a' defined here}}
// expected-error@+1 {{'ld_arg1a' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
void ld_arg1a(long double ld) {}

// TODO: We should diagnose the return type and argument type here.
typedef long double ld_ty;
// expected-note@+2 {{'ld_return1b' defined here}}
// expected-error@+1 {{'ld_return1b' requires 128 bit size 'ld_ty' (aka 'long double') type support, but device 'nvptx64-unknown-unknown' does not support it}}
ld_ty ld_return1b() { return 0; }
// expected-note@+2 {{'ld_arg1b' defined here}}
// expected-error@+1 {{'ld_arg1b' requires 128 bit size 'ld_ty' (aka 'long double') type support, but device 'nvptx64-unknown-unknown' does not support it}}
void ld_arg1b(ld_ty ld) {}

// TODO: These errors should not be emitted.
// expected-note@+2 {{'ld_return1c' defined here}}
// expected-error@+1 {{'ld_return1c' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
static long double ld_return1c() { return 0; }
// expected-note@+2 {{'ld_arg1c' defined here}}
// expected-error@+1 {{'ld_arg1c' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
static void ld_arg1c(long double ld) {}

// TODO: These errors should not be emitted.
// expected-note@+2 {{'ld_return1d' defined here}}
// expected-error@+1 {{'ld_return1d' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
inline long double ld_return1d() { return 0; }
// expected-note@+2 {{'ld_arg1d' defined here}}
// expected-error@+1 {{'ld_arg1d' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
inline void ld_arg1d(long double ld) {}

// expected-error@+2 {{'ld_return1e' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 {{'ld_return1e' defined here}}
static long double ld_return1e() { return 0; }
// expected-error@+2 {{'ld_arg1e' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 {{'ld_arg1e' defined here}}
static void ld_arg1e(long double ld) {}

// expected-error@+2 {{'ld_return1f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 {{'ld_return1f' defined here}}
inline long double ld_return1f() { return 0; }
// expected-error@+2 {{'ld_arg1f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
// expected-note@+1 {{'ld_arg1f' defined here}}
inline void ld_arg1f(long double ld) {}

inline void ld_use1() {
// expected-note@+1 {{'ld' defined here}}
  long double ld = 0;
// TODO: We should not diagnose this as the function is dead.
// expected-error@+1 {{'ld' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  ld += 1;
}
static void ld_use2() {
// expected-note@+1 {{'ld' defined here}}
  long double ld = 0;
// TODO: We should not diagnose this as the function is dead.
// expected-error@+1 {{'ld' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  ld += 1;
}

inline void ld_use3() {
// expected-note@+1 {{'ld' defined here}}
  long double ld = 0;
// expected-error@+1 {{'ld' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  ld += 1;
}
static void ld_use4() {
// expected-note@+1 {{'ld' defined here}}
  long double ld = 0;
// expected-error@+1 {{'ld' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  ld += 1;
}

void external() {
  void *p1 = reinterpret_cast<void*>(&ld_return1e);
  void *p2 = reinterpret_cast<void*>(&ld_arg1e);
  void *p3 = reinterpret_cast<void*>(&ld_return1f);
  void *p4 = reinterpret_cast<void*>(&ld_arg1f);
  void *p5 = reinterpret_cast<void*>(&ld_use3);
  void *p6 = reinterpret_cast<void*>(&ld_use4);
}

#ifndef _ARCH_PPC
// expected-note@+2 {{'ld_return2a' defined here}}
// expected-error@+1 {{'ld_return2a' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
__float128 ld_return2a() { return 0; }
// expected-note@+2 {{'ld_arg2a' defined here}}
// expected-error@+1 {{'ld_arg2a' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
void ld_arg2a(__float128 ld) {}

typedef __float128 fp128_ty;
// expected-note@+2 {{'ld_return2b' defined here}}
// expected-error@+1 {{'ld_return2b' requires 128 bit size 'fp128_ty' (aka '__float128') type support, but device 'nvptx64-unknown-unknown' does not support it}}
fp128_ty ld_return2b() { return 0; }
// expected-note@+2 {{'ld_arg2b' defined here}}
// expected-error@+1 {{'ld_arg2b' requires 128 bit size 'fp128_ty' (aka '__float128') type support, but device 'nvptx64-unknown-unknown' does not support it}}
void ld_arg2b(fp128_ty ld) {}
#endif

#pragma omp end declare target

// TODO: There should not be an error here, dead_inline is never emitted.
// expected-note@+1 {{'f' defined here}}
inline long double dead_inline(long double f) {
#pragma omp target map(f)
  // expected-error@+1 {{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}

// TODO: There should not be an error here, dead_static is never emitted.
// expected-note@+1 {{'f' defined here}}
static long double dead_static(long double f) {
#pragma omp target map(f)
  // expected-error@+1 {{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}

template<typename T>
long double dead_template(long double f) {
#pragma omp target map(f)
  f = 1;
  return f;
}

#ifndef _ARCH_PPC
// expected-note@+1 {{'f' defined here}}
__float128 foo2(__float128 f) {
#pragma omp target map(f)
  // expected-error@+1 {{'f' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}
#else
// expected-note@+1 {{'f' defined here}}
long double foo3(long double f) {
#pragma omp target map(f)
  // expected-error@+1 {{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}
#endif

T foo3() {
  T S;
#pragma omp target map(S)
  S.a = 1;
  return S;
}

// Allow all sorts of stuff on host
#ifndef _ARCH_PPC
__float128 q, b;
__float128 c = q + b;
#else
long double q, b;
long double c = q + b;
#endif

void hostFoo() {
  boo(c - b);
}

long double qa, qb;
decltype(qa + qb) qc;
double qd[sizeof(-(-(qc * 2)))];

struct A { };

template <bool>
struct A_type { typedef A type; };

template <class Sp, class Tp>
struct B {
  enum { value = bool(Sp::value) || bool(Tp::value) };
  typedef typename A_type<value>::type type;
};

void bar(_ExtInt(66) a) {
  auto b = a;
}
