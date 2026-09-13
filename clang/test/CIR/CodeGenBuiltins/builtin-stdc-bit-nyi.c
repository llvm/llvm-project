// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -std=c23 -fclangir -emit-cir -verify %s -o -

void test_unimplemented_builtin_stdc_bit(unsigned _BitInt(37) bi) {
  (void)__builtin_stdc_leading_zeros(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_count_zeros(bi);   // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_has_single_bit(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_bit_ceil(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_bit_floor(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_first_leading_zero(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_first_leading_one(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_first_trailing_zero(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
  (void)__builtin_stdc_first_trailing_one(bi); // expected-error {{ClangIR code gen Not Yet Implemented: stdc bit builtin with unsupported argument integer width}}
}
