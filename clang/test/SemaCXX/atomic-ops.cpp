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

typedef volatile int VI;
typedef const int CI;
extern VI vi;
extern CI ci;
void sink(int &&);

void unqualified_result() {
  sink(__atomic_load_n(&vi, __ATOMIC_RELAXED));
  sink(__atomic_load_n(&ci, __ATOMIC_RELAXED));
  sink(__atomic_exchange_n(&vi, 0, __ATOMIC_RELAXED));
}
