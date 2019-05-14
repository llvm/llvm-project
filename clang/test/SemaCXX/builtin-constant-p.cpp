// RUN: %clang_cc1 -std=c++17 -verify %s

using intptr_t = __INTPTR_TYPE__;

// Test interaction of constexpr and __builtin_constant_p.

template<typename T> constexpr bool bcp(T t) {
  return __builtin_constant_p(t);
}
template<typename T> constexpr bool bcp_fold(T t) {
  return __builtin_constant_p(((void)(intptr_t)&t, t));
}

constexpr intptr_t ensure_fold_is_generally_not_enabled = // expected-error {{constant expression}}
    (intptr_t)&ensure_fold_is_generally_not_enabled; // expected-note {{cast}}

constexpr intptr_t ptr_to_int(const void *p) {
  return __builtin_constant_p(1) ? (intptr_t)p : (intptr_t)p;
}

constexpr int *int_to_ptr(intptr_t n) {
  return __builtin_constant_p(1) ? (int*)n : (int*)n;
}

int x;

// Integer and floating point constants encountered during constant expression
// evaluation are considered constant. So is nullptr_t.
static_assert(bcp(1));
static_assert(bcp_fold(1));
static_assert(bcp(1.0));
static_assert(bcp_fold(1.0));
static_assert(bcp(nullptr));
static_assert(bcp_fold(nullptr));

// Pointers to the start of strings are considered constant.
static_assert(bcp("foo"));
static_assert(bcp_fold("foo"));

// Null pointers are considered constant.
static_assert(bcp<int*>(nullptr));
static_assert(bcp_fold<int*>(nullptr));
static_assert(bcp<const char*>(nullptr));
static_assert(bcp_fold<const char*>(nullptr));

// Other pointers are not.
static_assert(!bcp(&x));
static_assert(!bcp_fold(&x));

// Pointers cast to integers follow the rules for pointers.
static_assert(bcp(ptr_to_int("foo")));
static_assert(bcp_fold(ptr_to_int("foo")));
static_assert(!bcp(ptr_to_int(&x)));
static_assert(!bcp_fold(ptr_to_int(&x)));

// Integers cast to pointers follow the integer rules.
static_assert(bcp(int_to_ptr(0)));
static_assert(bcp_fold(int_to_ptr(0)));
static_assert(bcp(int_to_ptr(123)));      // GCC rejects these due to not recognizing
static_assert(bcp_fold(int_to_ptr(123))); // the bcp conditional in 'int_to_ptr' ...
static_assert(__builtin_constant_p((int*)123)); // ... but GCC accepts this
