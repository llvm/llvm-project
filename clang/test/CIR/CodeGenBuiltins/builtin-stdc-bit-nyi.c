// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -std=c23 -fclangir -emit-cir -verify %s -o -
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -isystem %S/Inputs -DTEST_LIB_SPELLINGS -fclangir -emit-cir -verify %s -o -

#ifdef TEST_LIB_SPELLINGS
#include <stdbit.h>
#endif

#ifndef TEST_LIB_SPELLINGS

void test_unimplemented_builtin_stdc_bit(unsigned int ui,
                                         unsigned _BitInt(37) bi) {
  volatile unsigned int r;
  r = __builtin_stdc_bit_ceil(ui);  // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: __builtin_stdc_bit_ceil}}
  r = __builtin_stdc_bit_floor(ui); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: __builtin_stdc_bit_floor}}
  (void)__builtin_stdc_leading_zeros(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_count_zeros(bi);   // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_has_single_bit(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
}

#else

void test_unimplemented_stdc_first(unsigned int x) {
  volatile unsigned int r;
  r = stdc_bit_ceil_ui(x);  // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_bit_ceil_ui}}
  r = stdc_bit_floor_ui(x); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_bit_floor_ui}}
  (void)stdc_first_leading_zero_ui(x);  // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_first_leading_zero_ui}}
  (void)stdc_first_leading_one_ui(x);   // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_first_leading_one_ui}}
  (void)stdc_first_trailing_zero_ui(x); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_first_trailing_zero_ui}}
  (void)stdc_first_trailing_one_ui(x);  // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: stdc_first_trailing_one_ui}}
}

#endif
