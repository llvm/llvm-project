// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-bit %t
// CHECK-FIXES: #include <bit>

/*
 * has_one_bit pattern
 */
unsigned has_one_bit_bithack(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned long has_one_bit_bithack(unsigned long x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned short has_one_bit_bithack(unsigned short x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned has_one_bit_bithack_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !((x - 1) & (x));
}

unsigned has_one_bit_bithack_otherperm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return !((x - 1) & (x)) && x;
}

unsigned has_one_bit_bithack_variant_neq(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned has_one_bit_bithack_variant_neq_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned has_one_bit_bithack_variant_gt(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}

unsigned has_one_bit_bithacks_variant_gte(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x >= 1) && !(x & (x - 1));
}

unsigned has_one_bit_bithacks_variant_lt(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (0 < x) && !(x & (x - 1));
}

unsigned has_one_bit_bithacks_variant_lte(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (1 <= x) && !(x & (x - 1));
}

unsigned has_one_bit_bithack_variant_gt_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}

#define HAS_ONE_BIT v && !(v & (v - 1))
unsigned has_one_bit_bithack_macro(unsigned v) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::has_one_bit' instead [modernize-use-std-bit]
  // No fixes, it comes from macro expansion.
  return HAS_ONE_BIT;
}

/*
 * Invalid has_one_bit patterns
 */
struct integer_like {
  integer_like operator!() const;
  bool operator&&(integer_like) const;
  integer_like operator&(integer_like) const;
  friend integer_like operator-(integer_like, unsigned);
};

unsigned invalid_has_one_bit_bithack(integer_like w, unsigned x, signed y, unsigned z) {
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
T has_one_bit_bithack_generic(T x) {
  // substitution only valid for some instantiation of has_one_bit_bithack_generic
  return x && !(x & (x - 1));
}

/*
 * popcount pattern
 */
namespace std {
using size_t = decltype(sizeof(0));
template<size_t N> class bitset {
  public:
  bitset(unsigned long);
  size_t count() const;
};
}

unsigned popcount_bitset(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(x);
  return std::bitset<sizeof(x) * 8>(x).count();
}

unsigned popcount_bitset_short(unsigned short x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(x);
  return std::bitset<sizeof(x) * 8>(x).count();
}

unsigned popcount_bitset_larger(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(x);
  return std::bitset<sizeof(x) * 16>(x).count();
}

unsigned popcount_bitset_uniform_init(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(x);
  return std::bitset<sizeof(x) * 16>{x}.count();
}

unsigned popcount_bitset_expr(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(x + 1);
  return std::bitset<sizeof(x) * 8>{x + 1}.count();
}

unsigned popcount_bitset_cast(unsigned x) {
  std::bitset<7>(static_cast<unsigned char>(x)).count(); // no warn
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::popcount(static_cast<unsigned char>(x));
  return std::bitset<8>(static_cast<unsigned char>(x)).count();
}

#define POPCOUNT std::bitset<sizeof(v) * 8>(static_cast<unsigned>(v)).count()
unsigned popcount_bitset_macro(unsigned v) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::popcount' instead [modernize-use-std-bit]
  // No fixes, it comes from macro expansion.
  return POPCOUNT;
}


/*
 * Invalid has_one_bit patterns
 */
template<std::size_t N> class bitset {
  public:
  bitset(unsigned long long);
  std::size_t count() const;
};

unsigned invalid_popcount_bitset(unsigned x, signed y) {
  std::size_t patterns[] = {
    // truncating bitset
    std::bitset<1>{x}.count(),
    // unsupported types
    std::bitset<sizeof(y) * 8>(y).count(),
    bitset<sizeof(x) * 8>{x}.count(),
  };
}

