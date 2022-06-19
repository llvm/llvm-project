// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux -fopenmp -fopenmp-version=51 -std=c++11 -o - %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux -fopenmp -fopenmp-version=51 -std=c++11 \
// RUN:  -DNO_INTEROP_T_DEF -o - %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux -fopenmp -fopenmp-version=50 -std=c++11 -o - %s
// RUN: %clang_cc1 -verify -triple x86_64-unknown-linux -fopenmp -fopenmp-version=51 -Wno-strict-prototypes -DC -x c -o - %s
// RUN: %clang_cc1 -verify -triple x86_64-pc-windows-msvc -fms-compatibility \
// RUN:  -fopenmp -fopenmp-version=51 -Wno-strict-prototypes -DC -DWIN -x c -o - %s

#ifdef NO_INTEROP_T_DEF
void foo_v1(float *, void *);
// expected-error@+1 {{'omp_interop_t' must be defined when 'append_args' clause is used; include <omp.h>}}
#pragma omp declare variant(foo_v1) append_args(interop(target)) \
  match(construct={dispatch})
void foo_v1(float *);
#else
typedef void *omp_interop_t;

int Other;

#if _OPENMP >= 202011  // At least OpenMP 5.1
#ifdef __cplusplus
class A {
public:
  void memberfoo_v0(float *A, float *B, int *I);
  void memberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp);

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (A::*)(float *, float *, int *)' is incompatible with type 'void (A::*)(float *, float *, int *, omp_interop_t)' with appended arguments}}
  #pragma omp declare variant(memberfoo_v0) match(construct={dispatch}) \
                        append_args(interop(target))

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (A::*)(float *, float *, int *, omp_interop_t)' is incompatible with type 'void (A::*)(float *, float *, int *, omp_interop_t, omp_interop_t)' with appended arguments}}
  #pragma omp declare variant(memberfoo_v1) match(construct={dispatch}) \
                        append_args(interop(target),interop(target))
  void memberbar(float *A, float *B, int *I) { return; }

  static void smemberfoo_v0(float *A, float *B, int *I);
  static void smemberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp);

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *)' is incompatible with type 'void (float *, float *, int *)' with appended arguments}}
  #pragma omp declare variant(smemberfoo_v0) match(construct={dispatch}) \
                        append_args(interop(target))

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *, omp_interop_t)' (aka 'void (float *, float *, int *, void *)') is incompatible with type 'void (float *, float *, int *)' with appended arguments}}
  #pragma omp declare variant(smemberfoo_v1) match(construct={dispatch}) \
                        append_args(interop(target),interop(target))
  static void smemberbar(float *A, float *B, int *I) { return; }

  virtual void vmemberfoo_v0(float *A, float *B, int *I);
  virtual void vmemberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp);

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (A::*)(float *, float *, int *)' is incompatible with type 'void (A::*)(float *, float *, int *, omp_interop_t)' with appended arguments}}
  #pragma omp declare variant(vmemberfoo_v0) match(construct={dispatch}) \
    append_args(interop(target))

  #pragma omp declare variant(vmemberfoo_v1) match(construct={dispatch}) \
    append_args(interop(target))

  // expected-error@+1 {{'#pragma omp declare variant' does not support virtual functions}}
  virtual void vmemberbar(float *A, float *B, int *I) { return; }

  virtual void pvmemberfoo_v0(float *A, float *B, int *I) = 0;
  virtual void pvmemberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp) = 0;

  // expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (A::*)(float *, float *, int *)' is incompatible with type 'void (A::*)(float *, float *, int *, omp_interop_t)' with appended arguments}}
  #pragma omp declare variant(pvmemberfoo_v0) match(construct={dispatch}) \
    append_args(interop(target))

  #pragma omp declare variant(pvmemberfoo_v1) match(construct={dispatch}) \
    append_args(interop(target))

  // expected-error@+1 {{'#pragma omp declare variant' does not support virtual functions}}
  virtual void pvmemberbar(float *A, float *B, int *I) = 0;
};

template <typename T> void templatefoo_v0(const T& t);
template <typename T> void templatefoo_v1(const T& t, omp_interop_t I);
template <typename T> void templatebar(const T& t) {}

// expected-error@+1 {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'void (const int &)' with appended arguments}}
#pragma omp declare variant(templatefoo_v0<int>) match(construct={dispatch}) \
                        append_args(interop(target))

// expected-error@+1 {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'void (const int &)' with appended arguments}}
#pragma omp declare variant(templatefoo_v1<int>) match(construct={dispatch}) \
                        append_args(interop(target),interop(target))
void templatebar(const int &t) {}
#endif // __cplusplus
#endif // _OPENMP >= 202011

void foo_v1(float *AAA, float *BBB, int *I) { return; }
void foo_v2(float *AAA, float *BBB, int *I) { return; }
void foo_v3(float *AAA, float *BBB, int *I) { return; }
void foo_v4(float *AAA, float *BBB, int *I, omp_interop_t IOp) { return; }

#if _OPENMP >= 202011 // At least OpenMP 5.1
void vararg_foo(const char *fmt, omp_interop_t it, ...);
// expected-error@+3 {{'append_args' is not allowed with varargs functions}}
#pragma omp declare variant(vararg_foo) match(construct={dispatch}) \
                                        append_args(interop(target))
void vararg_bar(const char *fmt, ...) { return; }

// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (const char *, omp_interop_t, ...)' (aka 'void (const char *, void *, ...)') is incompatible with type 'void (const char *)' with appended arguments}}
#pragma omp declare variant(vararg_foo) match(construct={dispatch}) \
                                        append_args(interop(target))
void vararg_bar2(const char *fmt) { return; }

// expected-error@+3 {{'adjust_arg' argument 'AAA' used in multiple clauses}}
#pragma omp declare variant(foo_v1)                          \
   match(construct={dispatch}, device={arch(arm)})           \
   adjust_args(need_device_ptr:AAA,BBB) adjust_args(need_device_ptr:AAA)

// expected-error@+3 {{'adjust_arg' argument 'AAA' used in multiple clauses}}
#pragma omp declare variant(foo_v1)                          \
   match(construct={dispatch}, device={arch(ppc)}),          \
   adjust_args(need_device_ptr:AAA) adjust_args(nothing:AAA)

// expected-error@+2 {{use of undeclared identifier 'J'}}
#pragma omp declare variant(foo_v1)                          \
   adjust_args(nothing:J)                                    \
   match(construct={dispatch}, device={arch(x86,x86_64)})

// expected-error@+2 {{expected reference to one of the parameters of function 'foo'}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:Other)                                \
   match(construct={dispatch}, device={arch(x86,x86_64)})

// expected-error@+2 {{'adjust_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:BBB) match(construct={target}, device={arch(arm)})

// expected-error@+2 {{'adjust_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v3)                          \
   adjust_args(nothing:BBB) match(device={arch(ppc)})

// expected-error@+1 {{expected 'match', 'adjust_args', or 'append_args' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo_v1)

// expected-error@+1 {{expected 'match', 'adjust_args', or 'append_args' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo_v1) other

// expected-error@+2 {{unexpected operation specified in 'append_args' clause, expected 'interop'}}
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
                                    append_args(foobar(target))

// expected-error@+2 {{directive '#pragma omp declare variant' cannot contain more than one 'append_args' clause}}
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
                                    append_args(interop(target)) \
                                    append_args(interop(targetsync))

// expected-error@+2 {{'append_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v4)                          \
                    append_args(interop(target)) match(construct={target})

// expected-error@+2 {{'append_args' clause requires 'dispatch' context selector}}
#pragma omp declare variant(foo_v4)                          \
                    match(construct={target}) append_args(interop(target))

// expected-warning@+2 {{interop type 'target' cannot be specified more than once}}
#pragma omp declare variant(foo_v4) match(construct={dispatch}) \
                                    append_args(interop(target,target))

// expected-warning@+2 {{interop type 'targetsync' cannot be specified more than once}}
#pragma omp declare variant(foo_v4) match(construct={dispatch}) \
                                    append_args(interop(targetsync,targetsync))

// expected-error@+2 {{expected interop type: 'target' and/or 'targetsync'}}
#pragma omp declare variant(foo_v4) match(construct={dispatch}) \
                                    append_args(interop())

// expected-error@+2 {{expected interop type: 'target' and/or 'targetsync'}}
#pragma omp declare variant(foo_v4) match(construct={dispatch}) \
                                    append_args(interop(somethingelse))

// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *)' is incompatible with type 'void (float *, float *, int *)' with appended arguments}}
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
                                    append_args(interop(target))

// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *)' is incompatible with type 'void (float *, float *, int *)' with appended arguments}}
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
                                    append_args(interop(target),interop(targetsync))

// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *, omp_interop_t)' (aka 'void (float *, float *, int *, void *)') is incompatible with type 'void (float *, float *, int *)' with appended arguments}}
#pragma omp declare variant(foo_v4) match(construct={dispatch}) \
                                    append_args(interop(target),interop(targetsync))

// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'void (float *, float *, int *, omp_interop_t)' (aka 'void (float *, float *, int *, void *)') is incompatible with type 'void (float *, float *, int *)'}}
#pragma omp declare variant(foo_v4) match(construct={dispatch})

#endif // _OPENMP >= 202011
#if _OPENMP < 202011  // OpenMP 5.0 or lower
// expected-error@+2 {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo_v1)                            \
   adjust_args(need_device_ptr:AAA) match(device={arch(arm)})
// expected-error@+2 {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo_v1)                            \
   append_args(interop(target)) match(device={arch(arm)})
#endif // _OPENMP < 202011

void foo(float *AAA, float *BBB, int *I) { return; }

#endif // NO_INTEROP_T_DEF

#ifdef C
void c_variant(omp_interop_t);
// expected-error@+3 {{function with '#pragma omp declare variant' must have a prototype when 'append_args' is used}}
#pragma omp declare variant(c_variant)                         \
   append_args(interop(target)) match(construct={dispatch})
void c_base() {}
#ifdef WIN
void _cdecl win_c_variant(omp_interop_t);
// expected-error@+3 {{function with '#pragma omp declare variant' must have a prototype when 'append_args' is used}}
#pragma omp declare variant(win_c_variant)                     \
   append_args(interop(target)) match(construct={dispatch})
void _cdecl win_c_base() {}
#endif // WIN
#endif
