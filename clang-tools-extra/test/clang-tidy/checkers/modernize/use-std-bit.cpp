// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-bit %t -check-suffixes=,NOPROMOTION
// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-bit %t -config="{CheckOptions: { modernize-use-std-bit.HonorIntPromotion: true }}" -check-suffixes=,PROMOTION
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


/*
 * rotate patterns
 */

using uint64_t = __UINT64_TYPE__;
using uint32_t = __UINT32_TYPE__;

int rotate_left_pattern(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+3]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return std::rotl(x, 3);
  // CHECK-FIXES-PROMOTION: return static_cast<int>(std::rotl(x, 3));
  return (x) << 3 | x >> 5;
}

auto rotate_left_pattern_with_cast(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:29: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES: return static_cast<short>(std::rotl(x, 3));
  return static_cast<short>((x) << 3 | x >> 5);
}

unsigned char rotate_left_pattern_with_implicit_cast(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+3]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return std::rotl(x, 3);
  // CHECK-FIXES-PROMOTION: return static_cast<int>(std::rotl(x, 3));
  return (x) << 3 | x >> 5;
}

auto rotate_left_pattern_without_cast(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+3]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return std::rotl(x, 3);
  // CHECK-FIXES-PROMOTION: return static_cast<int>(std::rotl(x, 3));
  return x << 3 | x >> 5;
}

uint32_t rotate_left_pattern_with_surrounding_parenthesis(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+3]]:11: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return (std::rotl(x, 3));
  // CHECK-FIXES-PROMOTION: return (static_cast<int>(std::rotl(x, 3)));
  return (x << 3 | x >> 5);
}

uint64_t rotate_left_pattern_int64(uint64_t x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::rotl(x, 3);
  return x << 3 | x >> 61;
}

uint32_t rotate_left_pattern_int32(uint32_t x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::rotl(x, 3);
  return (x) << 3 | x >> 29;
}

unsigned char rotate_left_pattern_perm(unsigned char x) {
  // CHECK-MESSAGES: :[[@LINE+3]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return std::rotl(x, 3);
  // CHECK-FIXES-PROMOTION: return static_cast<int>(std::rotl(x, 3));
  return x >> 5 | x << 3;
}

uint32_t rotate_swap_pattern(uint32_t x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::rotl' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::rotl(x, 16);
  return x << 16 | x >> 16;
}

uint64_t rotate_right_pattern(uint64_t x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::rotr' instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::rotr(x, 3);
  return (x << 61) | ((x >> 3));
}

unsigned char rotate_right_pattern_perm(unsigned char x0) {
  // CHECK-MESSAGES: :[[@LINE+3]]:10: warning: use 'std::rotr' instead [modernize-use-std-bit]
  // CHECK-FIXES-NOPROMOTION: return std::rotr(x0, 3);
  // CHECK-FIXES-PROMOTION: return static_cast<int>(std::rotr(x0, 3));
  return x0 >> 3 | x0 << 5;
}

#define ROTR v >> 3 | v << 5
unsigned char rotate_macro(unsigned char v) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use 'std::rotr' instead [modernize-use-std-bit]
  // No fixes, it comes from macro expansion.
  return ROTR;
}

/*
 * Invalid rotate patterns
 */
void invalid_rotate_patterns(unsigned char x, signed char y, unsigned char z) {
  int patterns[] = {
    // non-matching references
    x >> 3 | z << 5,
    // bad shift combination
    x >> 3 | x << 6,
    x >> 4 | x << 3,
    // bad operator combination
    x << 3 | x << 6,
    x + 3 | x << 6,
    x >> 3 & x << 5,
    x >> 5 ^ x << 3,
    // unsupported types
    y >> 4 | y << 4,
  };
}
