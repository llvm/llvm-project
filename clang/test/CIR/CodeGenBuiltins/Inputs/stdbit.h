#ifndef LLVM_CLANG_TEST_CIR_STDBIT_H
#define LLVM_CLANG_TEST_CIR_STDBIT_H

unsigned int stdc_leading_ones_uc(unsigned char);
unsigned int stdc_leading_ones_us(unsigned short);
unsigned int stdc_leading_ones_ui(unsigned int);
unsigned int stdc_leading_ones_ul(unsigned long);
unsigned int stdc_leading_ones_ull(unsigned long long);

unsigned int stdc_trailing_ones_uc(unsigned char);
unsigned int stdc_trailing_ones_us(unsigned short);
unsigned int stdc_trailing_ones_ui(unsigned int);
unsigned int stdc_trailing_ones_ul(unsigned long);
unsigned int stdc_trailing_ones_ull(unsigned long long);

unsigned int stdc_count_zeros_uc(unsigned char);
unsigned int stdc_count_zeros_us(unsigned short);
unsigned int stdc_count_zeros_ui(unsigned int);
unsigned int stdc_count_zeros_ul(unsigned long);
unsigned int stdc_count_zeros_ull(unsigned long long);

_Bool stdc_has_single_bit_uc(unsigned char);
_Bool stdc_has_single_bit_us(unsigned short);
_Bool stdc_has_single_bit_ui(unsigned int);
_Bool stdc_has_single_bit_ul(unsigned long);
_Bool stdc_has_single_bit_ull(unsigned long long);

unsigned int stdc_first_leading_zero_ui(unsigned int);
unsigned int stdc_first_leading_one_ui(unsigned int);
unsigned int stdc_first_trailing_zero_ui(unsigned int);
unsigned int stdc_first_trailing_one_ui(unsigned int);

unsigned int stdc_bit_width_uc(unsigned char);
unsigned int stdc_bit_width_us(unsigned short);
unsigned int stdc_bit_width_ui(unsigned int);
unsigned int stdc_bit_width_ul(unsigned long);
unsigned int stdc_bit_width_ull(unsigned long long);

unsigned int stdc_bit_ceil_ui(unsigned int);
unsigned int stdc_bit_floor_ui(unsigned int);

#endif
