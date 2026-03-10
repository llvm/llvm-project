// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-bit %t
// CHECK-FIXES: #include <bit>

unsigned bithacks(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned long bithacks(unsigned long x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned short bithacks(unsigned short x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned bithacks_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !((x - 1) & (x));
}

unsigned bithacks_otherperm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return !((x - 1) & (x)) && x;
}

unsigned bithacks_variant_neq(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned bithacks_variant_neq_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned bithacks_variant_gt(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}

unsigned bithacks_variant_gte(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x >= 1) && !(x & (x - 1));
}

unsigned bithacks_variant_lt(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (0 < x) && !(x & (x - 1));
}

unsigned bithacks_variant_lte(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (1 <= x) && !(x & (x - 1));
}

unsigned bithacks_variant_gt_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}

#define HAS_ONE_BIT v && !(v & (v - 1))
unsigned bithacks_macro(unsigned v) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // No fixes, it comes from macro expansion.
  return HAS_ONE_BIT;
}

/*
 * Invalid patterns
 */
struct integer_like {
  integer_like operator!() const;
  bool operator&&(integer_like) const;
  integer_like operator&(integer_like) const;
  friend integer_like operator-(integer_like, unsigned);
};

unsigned invalid_bithacks(integer_like w, unsigned x, signed y, unsigned z) {
  bool patterns[] = {
    // non commutative operators
    x && !(x & (1 - x)),
    x < 0 && !(x & (x - 1)),
    x >= 0 && !(x & (x - 1)),
    // unsupported combinations
    x && !(x & (z - 1)),
    z && !(x & (x - 1)),
    x && !(z & (x - 1)),
    // invalid operators
    x && !(x | (x - 1)),
    (bool)(x & !(x & (x - 1))),
    x && (x & (x - 1)),
    // unsupported types
    y && !(y & (y - 1)),
    w && !(w & (w - 1)),
  };
}

template <class T>
T bithacks_generic(T x) {
  // substitution only valid for some instantiation of bithacks_generic
  return x && !(x & (x - 1));
}
