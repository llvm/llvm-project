// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify -D __TEST_ELT_SI
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify -D __TEST_ELT_F
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify -D __TEST_ELT_SLL
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify -D __TEST_ELT_D
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify -D __TEST_UNALIGNED_UI
// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm -ferror-limit 10 %s -verify

#include <altivec.h>

vector signed int vsia;
vector unsigned int vuia;
vector signed long long vslla;
vector unsigned long long vulla;
vector float vfa;
vector double vda;
signed int sia;
unsigned int uia;
signed long long slla;
unsigned long long ulla;
float fa;
double da;

#ifdef __TEST_ELT_SI
vector signed int test_vec_replace_elt_si(void) {
  return vec_replace_elt(vsia, sia, 4); // expected-error {{element number 4 is outside of the valid range [0, 3]}}
}

#elif defined(__TEST_ELT_F)
vector float test_vec_replace_elt_f(void) {
  return vec_replace_elt(vfa, fa, 10); // expected-error {{element number 10 is outside of the valid range [0, 3]}}
}

#elif defined(__TEST_ELT_SLL)
vector signed long long test_vec_replace_elt_sll(void) {
  return vec_replace_elt(vslla, slla, 2); // expected-error {{element number 2 is outside of the valid range [0, 1]}}
}

#elif defined(__TEST_ELT_D)
vector double test_vec_replace_elt_d(void) {
  return vec_replace_elt(vda, da, 3); // expected-error {{element number 3 is outside of the valid range [0, 1]}}
}

#elif defined(__TEST_UNALIGNED_UI)
vector unsigned char test_vec_replace_unaligned_ui(void) {
  return vec_replace_unaligned(vuia, uia, 16); // expected-error {{byte number 16 is outside of the valid range [0, 12]}}
}

#else
vector unsigned char test_vec_replace_unaligned_ull(void) {
  return vec_replace_unaligned(vulla, ulla, 12); // expected-error {{byte number 12 is outside of the valid range [0, 8]}}
}
#endif
