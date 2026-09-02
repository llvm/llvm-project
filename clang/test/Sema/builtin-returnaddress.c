// RUN: %clang_cc1 -fsyntax-only -Wframe-address -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wmost -verify %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wframe-address -verify %s

__attribute__((noinline)) void* a(unsigned x) {
return __builtin_return_address(0);
}

__attribute__((noinline)) void* b(unsigned x) {
return __builtin_return_address(1); // expected-warning{{calling '__builtin_return_address' with a nonzero argument is unsafe}}
}

__attribute__((noinline)) void* c(unsigned x) {
return __builtin_frame_address(0);
}

__attribute__((noinline)) void* d(unsigned x) {
return __builtin_frame_address(1); // expected-warning{{calling '__builtin_frame_address' with a nonzero argument is unsafe}}
}

void* e(unsigned x) {
  return __builtin_frame_address(0); // expected-warning{{calling '__builtin_frame_address' in function not marked __attribute__((noinline)) may return a caller's frame address}}
}

void* f(unsigned x) {
  return __builtin_return_address(0); // expected-warning{{calling '__builtin_return_address' in function not marked __attribute__((noinline)) may return a caller's return address}}
}

#ifdef __cplusplus
template<int N> __attribute__((noinline)) void *RA()
{
  return __builtin_return_address(N); // expected-warning{{calling '__builtin_return_address' with a nonzero argument is unsafe}}
}

void *foo()
{
 return RA<2>(); // expected-note{{in instantiation of function template specialization 'RA<2>' requested here}}
}

void* f() {
  return ([&] () {
    return __builtin_frame_address(0); // expected-warning{{calling '__builtin_frame_address' in function not marked __attribute__((noinline)) may return a caller's frame address}}
  })();
}

void* g() {
  return ([&] () __attribute__((noinline)) {
    return __builtin_frame_address(0);
  })();
}

void* h() {
  return ([&] () {
    return __builtin_return_address(0); // expected-warning{{calling '__builtin_return_address' in function not marked __attribute__((noinline)) may return a caller's return address}}
  })();
}

void* i() {
  return ([&] () __attribute__((noinline)) {
    return __builtin_return_address(0);
  })();
}
#endif