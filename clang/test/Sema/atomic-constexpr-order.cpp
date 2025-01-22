// RUN: %clang_cc1 -std=c++2c -verify %s

// we can't use unknown ordering values
namespace load {

constexpr int constant = 42;
constexpr int both = __atomic_load_n(&constant, 111111); // #constant
// expected-note@#constant {{evaluated memory order argument to atomic operation is not allowed}}
// expected-warning@#constant {{memory order argument to atomic operation is invalid}}
// expected-error@#constant {{constexpr variable 'both' must be initialized by a constant expression}}

constexpr int load_with_order(int order) {
  int val = 42;
  return __atomic_load_n(&val, order); // #load_n
  // expected-note@#load_n 1+ {{evaluated memory order argument to atomic operation is not allowed}}
}

// unknown values are not allowed
constexpr int tmp1 = load_with_order(111111); // #load-order-fail1
// expected-error-re@#load-order-fail1 {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}} 
// expected-note-re@#load-order-fail1 {{in call to 'load_with_order({{[0-9]+}})'}}
constexpr int tmp2 = load_with_order(256); // #load-order-fail2
// expected-error-re@#load-order-fail2 {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}} 
// expected-note-re@#load-order-fail2 {{in call to 'load_with_order({{[0-9]+}})'}}

// load ordering allowed
constexpr int tmp3 = load_with_order(__ATOMIC_RELAXED); // fine
constexpr int tmp4 = load_with_order(__ATOMIC_CONSUME); // fine
constexpr int tmp5 = load_with_order(__ATOMIC_ACQUIRE); // fine
constexpr int tmp6 = load_with_order(__ATOMIC_SEQ_CST); // fine

// RELEASE or ACQ_REL are not allowed for load
constexpr int tmp7 = load_with_order(__ATOMIC_RELEASE); // #load-release-not-allowed
constexpr int tmp8 = load_with_order(__ATOMIC_ACQ_REL); // #load-acq-rel-not-allowed

// expected-error-re@#load-release-not-allowed {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#load-release-not-allowed  {{in call to 'load_with_order({{[0-9]+}})'}}

// expected-error-re@#load-acq-rel-not-allowed {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#load-acq-rel-not-allowed  {{in call to 'load_with_order({{[0-9]+}})'}}

}

namespace store {

constexpr bool store_with_order(int order) {
  int value = 0;
  __atomic_store_n(&value, 42, order); // #store
  // expected-note@#store 1+ {{evaluated memory order argument to atomic operation is not allowed}}
  return true;
}

// unknown values are not allowed
constexpr auto tmp1 = store_with_order(111111); // #store-order-fail1
// expected-error-re@#store-order-fail1 {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}} 
// expected-note-re@#store-order-fail1 {{in call to 'store_with_order({{[0-9]+}})'}}
constexpr auto tmp2 = store_with_order(256); // #store-order-fail2
// expected-error-re@#store-order-fail2 {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}} 
// expected-note-re@#store-order-fail2 {{in call to 'store_with_order({{[0-9]+}})'}}

constexpr int tmp3 = store_with_order(__ATOMIC_RELAXED); // fine
constexpr int tmp4 = store_with_order(__ATOMIC_CONSUME); // #store-consume-not-allowed
constexpr int tmp5 = store_with_order(__ATOMIC_ACQUIRE); // #store-acquire-not-allowed
constexpr int tmp6 = store_with_order(__ATOMIC_SEQ_CST); // fine
constexpr int tmp7 = store_with_order(__ATOMIC_RELEASE); // fine
constexpr int tmp8 = store_with_order(__ATOMIC_ACQ_REL); // #store-acq-rel-not-allowed

// expected-error-re@#store-consume-not-allowed {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#store-consume-not-allowed  {{in call to 'store_with_order({{[0-9]+}})'}}

// expected-error-re@#store-acquire-not-allowed {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#store-acquire-not-allowed  {{in call to 'store_with_order({{[0-9]+}})'}}

// expected-error-re@#store-acq-rel-not-allowed {{constexpr variable '{{[a-z0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#store-acq-rel-not-allowed  {{in call to 'store_with_order({{[0-9]+}})'}}



}

