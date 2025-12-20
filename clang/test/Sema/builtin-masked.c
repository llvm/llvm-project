// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Bool v8b __attribute__((ext_vector_type(8)));
typedef _Bool v2b __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

void test_masked_load(int *pf, v8b mask, v2b mask2, v2b thru) {
  (void)__builtin_masked_load(mask); // expected-error {{too few arguments to function call, expected 2, have 1}}
  (void)__builtin_masked_load(mask, pf, pf, pf); // expected-error {{too many arguments to function call, expected at most 3, have 4}}
  (void)__builtin_masked_load(mask, mask); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_load(mask2, pf, thru); // expected-error {{3rd argument must be a 'int __attribute__((ext_vector_type(2)))' (vector of 2 'int' values)}}
}

void test_masked_store(int *pf, v8f *pf2, v8b mask, v2b mask2) {
  __builtin_masked_store(mask); // expected-error {{too few arguments to function call, expected 3, have 1}}
  __builtin_masked_store(mask, 0, 0, 0); // expected-error {{too many arguments to function call, expected 3, have 4}}
  __builtin_masked_store(0, 0, pf); // expected-error {{1st argument must be a vector of boolean types (was 'int')}}
  __builtin_masked_store(mask, 0, pf); // expected-error {{2nd argument must be a vector}}
  __builtin_masked_store(mask, *pf, 0); // expected-error {{3rd argument must be a scalar pointer}}
}

void test_masked_expand_load(int *pf, v8b mask, v2b mask2, v2b thru) {
  (void)__builtin_masked_expand_load(mask); // expected-error {{too few arguments to function call, expected 2, have 1}}
  (void)__builtin_masked_expand_load(mask, pf, pf, pf); // expected-error {{too many arguments to function call, expected at most 3, have 4}}
  (void)__builtin_masked_expand_load(mask, mask); // expected-error {{2nd argument must be a scalar pointer}}
  (void)__builtin_masked_expand_load(mask2, pf, thru); // expected-error {{3rd argument must be a 'int __attribute__((ext_vector_type(2)))' (vector of 2 'int' values)}}
}

void test_masked_compress_store(int *pf, v8f *pf2, v8b mask, v2b mask2) {
  __builtin_masked_compress_store(mask); // expected-error {{too few arguments to function call, expected 3, have 1}}
  __builtin_masked_compress_store(mask, 0, 0, 0); // expected-error {{too many arguments to function call, expected 3, have 4}}
  __builtin_masked_compress_store(0, 0, pf); // expected-error {{1st argument must be a vector of boolean types (was 'int')}}
  __builtin_masked_compress_store(mask, 0, pf); // expected-error {{2nd argument must be a vector}}
  __builtin_masked_compress_store(mask, *pf, 0); // expected-error {{3rd argument must be a scalar pointer}}
}

void test_masked_gather(int *p, v8i idx, v8b mask, v2b mask2, v2b thru) {
  __builtin_masked_gather(mask); // expected-error {{too few arguments to function call, expected 3, have 1}}
  __builtin_masked_gather(mask, p, p, p, p, p); // expected-error {{too many arguments to function call, expected at most 4, have 6}}
  __builtin_masked_gather(p, p, p); // expected-error {{1st argument must be a vector of boolean types (was 'int *')}}
  __builtin_masked_gather(mask, p, p); // expected-error {{1st argument must be a vector of integer types (was 'int *')}}
  __builtin_masked_gather(mask2, idx, p); // expected-error {{all arguments to '__builtin_masked_gather' must have the same number of elements (was 'v2b'}}
  __builtin_masked_gather(mask, idx, p, thru); // expected-error {{4th argument must be a 'int __attribute__((ext_vector_type(8)))' (vector of 8 'int' values)}}
  __builtin_masked_gather(mask, idx, &idx); // expected-error {{3rd argument must be a scalar pointer}}
}

void test_masked_scatter(int *p, v8i idx, v8b mask, v2b mask2, v8i val) {
  __builtin_masked_scatter(mask); // expected-error {{too few arguments to function call, expected 4, have 1}}
  __builtin_masked_scatter(mask, p, p, p, p, p); // expected-error {{too many arguments to function call, expected 4, have 6}}
  __builtin_masked_scatter(p, p, p, p); // expected-error {{1st argument must be a vector of boolean types (was 'int *')}}
  __builtin_masked_scatter(mask, p, p, p); // expected-error {{2nd argument must be a vector of integer types (was 'int *')}}
  __builtin_masked_scatter(mask, idx, mask, p); // expected-error {{last two arguments to '__builtin_masked_scatter' must have the same type}}
  __builtin_masked_scatter(mask, idx, val, idx); // expected-error {{4th argument must be a scalar pointer}}
  __builtin_masked_scatter(mask, idx, val, &idx); // expected-error {{4th argument must be a scalar pointer}}
}

void a(v8b mask, v8i v, const int *ptr) {
  __builtin_masked_load(mask, ptr, v);
  (void)__builtin_masked_load(mask, (volatile int *)ptr, v); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
}

void b(v8b mask, v8i idx, const int *ptr) {
  (void)__builtin_masked_gather(mask, idx, ptr);
  (void)__builtin_masked_gather(mask, idx, (volatile int *)ptr); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
}

void c(v8b mask, const v8i v, int *ptr) {
  __builtin_masked_store(mask, v, ptr);
}

void readonly(v8b mask, v8i v, const int *ptr, const int *s) {
  (void)__builtin_masked_store(mask, v, ptr); // expected-error {{sending 'const int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('const int *' vs 'int *')}}
  (void)__builtin_masked_compress_store(mask, v, ptr); // expected-error {{sending 'const int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('const int *' vs 'int *')}}
  (void)__builtin_masked_scatter(mask, v, v, s); // expected-error {{sending 'const int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('const int *' vs 'int *')}}
}

void vol(v8b mask, v8i v, volatile int *ptr, volatile int *s) {
  (void)__builtin_masked_load(mask, ptr); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
  (void)__builtin_masked_store(mask, v, ptr); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
  (void)__builtin_masked_expand_load(mask, ptr); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
  (void)__builtin_masked_compress_store(mask, v, ptr); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
  (void)__builtin_masked_gather(mask, v, ptr);// expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
  (void)__builtin_masked_scatter(mask, v, v, s); // expected-error {{sending 'volatile int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('volatile int *' vs 'int *')}}
}

void as(v8b mask, int [[clang::address_space(999)]] * ptr, v8i v) {
  (void)__builtin_masked_load(mask, ptr);
  (void)__builtin_masked_store(mask, v, ptr);
  (void)__builtin_masked_expand_load(mask, ptr); // expected-error {{sending '__attribute__((address_space(999))) int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('__attribute__((address_space(999))) int *' vs 'int *')}}
  (void)__builtin_masked_compress_store(mask, v, ptr); // expected-error {{sending '__attribute__((address_space(999))) int *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('__attribute__((address_space(999))) int *' vs 'int *')}}
  (void)__builtin_masked_gather(mask, v, ptr);
  (void)__builtin_masked_scatter(mask, v, v, ptr);
}

void atom(v8b mask, _Atomic int * ptr, v8i v) {
  (void)__builtin_masked_load(mask, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
  (void)__builtin_masked_store(mask, v, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
  (void)__builtin_masked_expand_load(mask, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
  (void)__builtin_masked_compress_store(mask, v, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
  (void)__builtin_masked_gather(mask, v, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
  (void)__builtin_masked_scatter(mask, v, v, ptr); // expected-error {{'_Atomic(int) *' to parameter of incompatible type 'int *': type mismatch at 2nd parameter ('_Atomic(int) *' vs 'int *')}}
}
