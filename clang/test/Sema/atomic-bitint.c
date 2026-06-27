// RUN: %clang_cc1 %s -fsyntax-only -verify -triple x86_64-unknown-linux-gnu -std=c23
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple x86_64-unknown-linux-gnu -std=c2y
// RUN: %clang_cc1 %s -fsyntax-only -verify -triple riscv64-unknown-linux-gnu -std=c23
//
// C23 requires the type-generic atomic interfaces to accept _BitInt(N) for
// every N, so _Atomic(_BitInt(N)) is well-formed at every width. The atomic
// code imposes no width cap of its own; widths past 128 are available wherever
// the target accepts _BitInt > 128 (x86 and RISC-V today).

_Atomic(_BitInt(4))    a4;     // small
_Atomic(_BitInt(9))    a9;     // non-power-of-two
_Atomic(_BitInt(37))   a37;    // padded
_Atomic(_BitInt(64))   a64;
_Atomic(_BitInt(128))  a128;
_Atomic(_BitInt(256))  a256;   // wider than any inline atomic
_Atomic(_BitInt(4096)) a4096;  // far past the inline range

// The _Atomic qualifier spelling is equally valid.
_Atomic _BitInt(9) q9;

static_assert(sizeof(_Atomic(_BitInt(37))) == 8);
static_assert(sizeof(_Atomic(_BitInt(128))) == 16);
static_assert(sizeof(_Atomic(_BitInt(256))) == 32);

void c11_builtins(_Atomic(_BitInt(37)) *p, _BitInt(37) v, _BitInt(37) *e) {
  (void)__c11_atomic_load(p, __ATOMIC_SEQ_CST);
  __c11_atomic_store(p, v, __ATOMIC_SEQ_CST);
  (void)__c11_atomic_exchange(p, v, __ATOMIC_SEQ_CST);
  (void)__c11_atomic_compare_exchange_strong(p, e, v, __ATOMIC_SEQ_CST,
                                             __ATOMIC_SEQ_CST);
  (void)__c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
  (void)__c11_atomic_fetch_and(p, v, __ATOMIC_SEQ_CST);
  (void)__c11_atomic_fetch_min(p, v, __ATOMIC_SEQ_CST);
}

// The GNU __atomic_* builtins take a plain _BitInt pointer; the _fetch forms
// return the new value.
void gnu_builtins(_BitInt(37) *p, _BitInt(37) v) {
  (void)__atomic_load_n(p, __ATOMIC_SEQ_CST);
  __atomic_store_n(p, v, __ATOMIC_SEQ_CST);
  (void)__atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
  (void)__atomic_add_fetch(p, v, __ATOMIC_SEQ_CST);
}

// Lifting the _BitInt rejection must not lose the atomic-specific checks.
void rejects(_Atomic(_BitInt(37)) *ap, _BitInt(37) *p, _BitInt(37) v) {
  (void)__c11_atomic_load(ap); // expected-error {{too few arguments to function call}}
  (void)__c11_atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); // expected-error {{must be a pointer to _Atomic}}
}
struct WithAtomicBitIntField {
  _Atomic(_BitInt(5)) f : 3; // expected-error {{bit-field 'f' has non-integral type}}
};
