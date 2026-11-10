#ifndef LLVM_CLANG_TEST_CIR_STDBIT_H
#define LLVM_CLANG_TEST_CIR_STDBIT_H

unsigned int stdc_leading_zeros_uc(unsigned char);
unsigned int stdc_leading_zeros_us(unsigned short);
unsigned int stdc_leading_zeros_ui(unsigned int);
unsigned int stdc_leading_zeros_ul(unsigned long);
unsigned int stdc_leading_zeros_ull(unsigned long long);

unsigned int stdc_leading_ones_uc(unsigned char);
unsigned int stdc_leading_ones_us(unsigned short);
unsigned int stdc_leading_ones_ui(unsigned int);
unsigned int stdc_leading_ones_ul(unsigned long);
unsigned int stdc_leading_ones_ull(unsigned long long);

unsigned int stdc_trailing_zeros_uc(unsigned char);
unsigned int stdc_trailing_zeros_us(unsigned short);
unsigned int stdc_trailing_zeros_ui(unsigned int);
unsigned int stdc_trailing_zeros_ul(unsigned long);
unsigned int stdc_trailing_zeros_ull(unsigned long long);

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

unsigned int stdc_count_ones_uc(unsigned char);
unsigned int stdc_count_ones_us(unsigned short);
unsigned int stdc_count_ones_ui(unsigned int);
unsigned int stdc_count_ones_ul(unsigned long);
unsigned int stdc_count_ones_ull(unsigned long long);

_Bool stdc_has_single_bit_uc(unsigned char);
_Bool stdc_has_single_bit_us(unsigned short);
_Bool stdc_has_single_bit_ui(unsigned int);
_Bool stdc_has_single_bit_ul(unsigned long);
_Bool stdc_has_single_bit_ull(unsigned long long);

unsigned int stdc_first_leading_zero_uc(unsigned char);
unsigned int stdc_first_leading_zero_us(unsigned short);
unsigned int stdc_first_leading_zero_ui(unsigned int);
unsigned int stdc_first_leading_zero_ul(unsigned long);
unsigned int stdc_first_leading_zero_ull(unsigned long long);

unsigned int stdc_first_leading_one_uc(unsigned char);
unsigned int stdc_first_leading_one_us(unsigned short);
unsigned int stdc_first_leading_one_ui(unsigned int);
unsigned int stdc_first_leading_one_ul(unsigned long);
unsigned int stdc_first_leading_one_ull(unsigned long long);

unsigned int stdc_first_trailing_zero_uc(unsigned char);
unsigned int stdc_first_trailing_zero_us(unsigned short);
unsigned int stdc_first_trailing_zero_ui(unsigned int);
unsigned int stdc_first_trailing_zero_ul(unsigned long);
unsigned int stdc_first_trailing_zero_ull(unsigned long long);

unsigned int stdc_first_trailing_one_uc(unsigned char);
unsigned int stdc_first_trailing_one_us(unsigned short);
unsigned int stdc_first_trailing_one_ui(unsigned int);
unsigned int stdc_first_trailing_one_ul(unsigned long);
unsigned int stdc_first_trailing_one_ull(unsigned long long);

unsigned int stdc_bit_width_uc(unsigned char);
unsigned int stdc_bit_width_us(unsigned short);
unsigned int stdc_bit_width_ui(unsigned int);
unsigned int stdc_bit_width_ul(unsigned long);
unsigned int stdc_bit_width_ull(unsigned long long);

unsigned char stdc_bit_ceil_uc(unsigned char);
unsigned short stdc_bit_ceil_us(unsigned short);
unsigned int stdc_bit_ceil_ui(unsigned int);
unsigned long stdc_bit_ceil_ul(unsigned long);
unsigned long long stdc_bit_ceil_ull(unsigned long long);

unsigned char stdc_bit_floor_uc(unsigned char);
unsigned short stdc_bit_floor_us(unsigned short);
unsigned int stdc_bit_floor_ui(unsigned int);
unsigned long stdc_bit_floor_ul(unsigned long);
unsigned long long stdc_bit_floor_ull(unsigned long long);

#endif
