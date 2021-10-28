// RUN: %clang_cc1 %s -verify -ffreestanding -fsyntax-only -triple=i686-linux-gnu -std=c11 -Watomic-implicit-seq-cst

// _Atomic operations are implicitly sequentially-consistent. Some codebases
// want to force explicit usage of memory order instead.

_Atomic(int) atom;
void gimme_int(int);

void bad_pre_inc(void) {
  ++atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_pre_dec(void) {
  --atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_post_inc(void) {
  atom++; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_post_dec(void) {
  atom--; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_call(void) {
  gimme_int(atom); // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_unary_plus(void) {
  return +atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_unary_minus(void) {
  return -atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_unary_logical_not(void) {
  return !atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_unary_bitwise_not(void) {
  return ~atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_explicit_cast(void) {
  return (int)atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_implicit_cast(void) {
  return atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_mul_1(int i) {
  return atom * i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_mul_2(int i) {
  return i * atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_div_1(int i) {
  return atom / i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_div_2(int i) {
  return i / atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_mod_1(int i) {
  return atom % i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_mod_2(int i) {
  return i % atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_add_1(int i) {
  return atom + i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_add_2(int i) {
  return i + atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_sub_1(int i) {
  return atom - i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_sub_2(int i) {
  return i - atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_shl_1(int i) {
  return atom << i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_shl_2(int i) {
  return i << atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_shr_1(int i) {
  return atom >> i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_shr_2(int i) {
  return i >> atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_lt_1(int i) {
  return atom < i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_lt_2(int i) {
  return i < atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_le_1(int i) {
  return atom <= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_le_2(int i) {
  return i <= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_gt_1(int i) {
  return atom > i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_gt_2(int i) {
  return i > atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ge_1(int i) {
  return atom >= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ge_2(int i) {
  return i >= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_eq_1(int i) {
  return atom == i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_eq_2(int i) {
  return i == atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ne_1(int i) {
  return atom != i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ne_2(int i) {
  return i != atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitand_1(int i) {
  return atom & i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitand_2(int i) {
  return i & atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitxor_1(int i) {
  return atom ^ i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitxor_2(int i) {
  return i ^ atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitor_1(int i) {
  return atom | i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitor_2(int i) {
  return i | atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitnand_1(int i) {
  return ~(atom & i); // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_bitnand_2(int i) {
  return ~(i & atom); // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_and_1(int i) {
  return atom && i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_and_2(int i) {
  return i && atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_or_1(int i) {
  return atom || i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_or_2(int i) {
  return i || atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}
int bad_ternary_1(int i, int j) {
  return i ? atom : j; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ternary_2(int i, int j) {
  return atom ? i : j; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_ternary_3(int i, int j) {
  return i ? j : atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_assign_1(int i) {
  atom = i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_assign_2(int *i) {
  *i = atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_assign_3() {
  atom = atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_add_1(int i) {
  atom += i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_add_2(int *i) {
  *i += atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_sub_1(int i) {
  atom -= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_sub_2(int *i) {
  *i -= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_mul_1(int i) {
  atom *= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_mul_2(int *i) {
  *i *= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_div_1(int i) {
  atom /= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_div_2(int *i) {
  *i /= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_mod_1(int i) {
  atom %= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_mod_2(int *i) {
  *i %= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_shl_1(int i) {
  atom <<= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_shl_2(int *i) {
  *i <<= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_shr_1(int i) {
  atom >>= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_shr_2(int *i) {
  *i >>= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitand_1(int i) {
  atom &= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitand_2(int *i) {
  *i &= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitxor_1(int i) {
  atom ^= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitxor_2(int *i) {
  *i ^= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitor_1(int i) {
  atom |= i; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void bad_compound_bitor_2(int *i) {
  *i |= atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

int bad_comma(int i) {
  return (void)i, atom; // expected-warning {{implicit use of sequentially-consistent atomic may incur stronger memory barriers than necessary}}
}

void good_c11_atomic_init(int i) { __c11_atomic_init(&atom, i); }
void good_c11_atomic_thread_fence(void) { __c11_atomic_thread_fence(__ATOMIC_RELAXED); }
void good_c11_atomic_signal_fence(void) { __c11_atomic_signal_fence(__ATOMIC_RELAXED); }
void good_c11_atomic_is_lock_free(void) { __c11_atomic_is_lock_free(sizeof(int)); }
void good_c11_atomic_store(int i) { __c11_atomic_store(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_load(void) { return __c11_atomic_load(&atom, __ATOMIC_RELAXED); }
int good_c11_atomic_exchange(int i) { return __c11_atomic_exchange(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_compare_exchange_strong(int *e, int i) { return __c11_atomic_compare_exchange_strong(&atom, e, i, __ATOMIC_RELAXED, __ATOMIC_RELAXED); }
int good_c11_atomic_compare_exchange_weak(int *e, int i) { return __c11_atomic_compare_exchange_weak(&atom, e, i, __ATOMIC_RELAXED, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_add(int i) { return __c11_atomic_fetch_add(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_sub(int i) { return __c11_atomic_fetch_sub(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_and(int i) { return __c11_atomic_fetch_and(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_or(int i) { return __c11_atomic_fetch_or(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_xor(int i) { return __c11_atomic_fetch_xor(&atom, i, __ATOMIC_RELAXED); }
int good_c11_atomic_fetch_nand(int i) { return __c11_atomic_fetch_nand(&atom, i, __ATOMIC_RELAXED); }

void good_cast_to_void(void) { (void)atom; }
_Atomic(int) * good_address_of(void) { return &atom; }
int good_sizeof(void) { return sizeof(atom); }
_Atomic(int) * good_pointer_arith(_Atomic(int) * p) { return p + 10; }
_Bool good_pointer_to_bool(_Atomic(int) * p) { return p; }
void good_no_init(void) { _Atomic(int) no_init; }
void good_init(void) { _Atomic(int) init = 42; }
