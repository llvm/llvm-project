// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -verify=no256,no512 -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx -verify=no512 -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512f -verify=both -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512f -target-feature +evex512 -verify=both -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512f -target-feature -evex512 -verify=avx512-256 -DAVX512_ERR=1 -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512f -target-feature -evex512 -verify=avx512-256 -DAVX512_ERR=2 -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +avx512f -target-feature -evex512 -verify=avx512-256 -DAVX512_ERR=3 -o - -S
// REQUIRES: x86-registered-target

// both-no-diagnostics

typedef short avx512fType __attribute__((vector_size(64)));
typedef short avx256Type __attribute__((vector_size(32)));

__attribute__((target("avx"))) void takesAvx256(avx256Type t);
__attribute__((target("avx512f"))) void takesAvx512(avx512fType t);
__attribute__((target("avx512f,evex512"))) void takesAvx512_2(avx512fType t);
void takesAvx256_no_target(avx256Type t);
void takesAvx512_no_target(avx512fType t);

void variadic(int i, ...);
__attribute__((target("avx512f"))) void variadic_err(int i, ...);

#if !defined(AVX512_ERR) || AVX512_ERR == 1
// If neither side has an attribute, warn.
void call_warn(void) {
  avx256Type t1;
  takesAvx256_no_target(t1); // no256-warning {{AVX vector argument of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}

  avx512fType t2;
  // avx512-256-error@+1 {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'evex512' enabled changes the ABI}}
  takesAvx512_no_target(t2); // no512-warning {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'avx512f' enabled changes the ABI}}

  variadic(1, t1); // no256-warning {{AVX vector argument of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}
  // avx512-256-error@+1 {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'evex512' enabled changes the ABI}}
  variadic(3, t2); // no512-warning {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'avx512f' enabled changes the ABI}}
}
#endif

#if !defined(AVX512_ERR) || AVX512_ERR == 2
// If only 1 side has an attribute, error.
void call_errors(void) {
  avx256Type t1;
  takesAvx256(t1); // no256-error {{AVX vector argument of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}
  avx512fType t2;
  // avx512-256-error@+1 {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'evex512' enabled changes the ABI}}
  takesAvx512(t2); // no512-error {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'avx512f' enabled changes the ABI}}

  variadic_err(1, t1); // no256-error {{AVX vector argument of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}
  // avx512-256-error@+1 {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'evex512' enabled changes the ABI}}
  variadic_err(3, t2); // no512-error {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'avx512f' enabled changes the ABI}}
}
#endif

#if !defined(AVX512_ERR) || AVX512_ERR == 3
__attribute__((target("avx"))) void call_avx256_ok(void) {
  avx256Type t;
  takesAvx256(t);
}

// Option -mno-evex512 affects target attributes. To retain the 512-bit capability, an explict "evex512" must be added together.
__attribute__((target("avx512f,evex512"))) void call_avx512_ok1(void) {
  avx512fType t;
  takesAvx512_2(t);
}

__attribute__((target("avx512f"))) void call_avx512_ok2(void) {
  avx512fType t;
  takesAvx512(t); // avx512-256-error {{AVX vector argument of type 'avx512fType' (vector of 32 'short' values) without 'evex512' enabled changes the ABI}}
}
#endif
