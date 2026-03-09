// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-std-bit %t
// CHECK-FIXES: #include <bit>

unsigned bithacks(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned long bithacks(unsigned long x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned short bithacks(unsigned short x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !(x & (x - 1));
}

unsigned bithacks_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return x && !((x - 1) & (x));
}

unsigned bithacks_variant_neq(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned bithacks_variant_neq_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x != 0) && !(x & (x - 1));
}

unsigned bithacks_variant_gt(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}

unsigned bithacks_variant_gt_perm(unsigned x) {
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: use std::has_one_bit instead [modernize-use-std-bit]
  // CHECK-FIXES: return std::has_one_bit(x);
  return (x > 0) && !(x & (x - 1));
}
