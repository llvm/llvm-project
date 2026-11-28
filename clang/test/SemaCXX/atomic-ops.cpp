// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=i686-linux-gnu -std=c++11

// We crashed when we couldn't properly convert the first arg of __atomic_* to
// an lvalue.
void PR28623() {
  void helper(int); // expected-note{{target}}
  void helper(char); // expected-note{{target}}
  __atomic_store_n(helper, 0, 0); // expected-error{{reference to overloaded function could not be resolved}}
}

template<typename>
struct X {
  char arr[1];
};

extern X<void>* p, *q;

// They should be accepted.
void f() {
  __atomic_exchange(p, p, q, __ATOMIC_RELAXED);
  __atomic_load(p, p, __ATOMIC_RELAXED);
  __atomic_store(p, p, __ATOMIC_RELAXED);
  __atomic_compare_exchange(p, p, q, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}

using nullptr_t = decltype(nullptr);

void atomic_minmax_nullptr_t(_Atomic(nullptr_t)* an) {
  *an = __c11_atomic_fetch_max(an, nullptr, 0); // expected-error{{address argument to atomic operation must be a pointer to atomic integer, pointer or supported floating point type ('_Atomic(nullptr_t) *' invalid)}}
  *an = __c11_atomic_fetch_min(an, nullptr, 0); // expected-error{{address argument to atomic operation must be a pointer to atomic integer, pointer or supported floating point type ('_Atomic(nullptr_t) *' invalid)}}

}
