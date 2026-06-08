#ifndef LLVM_CLANG_TEST_STDBIT_H
#define LLVM_CLANG_TEST_STDBIT_H

#define stdc_leading_zeros(x) (__builtin_stdc_leading_zeros((x)))
#define stdc_leading_ones(x) (__builtin_stdc_leading_ones((x)))
#define stdc_trailing_zeros(x) (__builtin_stdc_trailing_zeros((x)))
#define stdc_trailing_ones(x) (__builtin_stdc_trailing_ones((x)))
#define stdc_first_leading_zero(x) (__builtin_stdc_first_leading_zero((x)))
#define stdc_first_leading_one(x) (__builtin_stdc_first_leading_one((x)))
#define stdc_first_trailing_zero(x) (__builtin_stdc_first_trailing_zero((x)))
#define stdc_first_trailing_one(x) (__builtin_stdc_first_trailing_one((x)))
#define stdc_count_zeros(x) (__builtin_stdc_count_zeros((x)))
#define stdc_count_ones(x) (__builtin_stdc_count_ones((x)))
#define stdc_has_single_bit(x) ((_Bool)__builtin_stdc_has_single_bit((x)))
#define stdc_bit_width(x) (__builtin_stdc_bit_width((x)))
#define stdc_bit_floor(x) (__builtin_stdc_bit_floor((x)))
#define stdc_bit_ceil(x) (__builtin_stdc_bit_ceil((x)))

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

unsigned int stdc_bit_width_uc(unsigned char);
unsigned int stdc_bit_width_us(unsigned short);
unsigned int stdc_bit_width_ui(unsigned int);
unsigned int stdc_bit_width_ul(unsigned long);
unsigned int stdc_bit_width_ull(unsigned long long);

unsigned char stdc_bit_floor_uc(unsigned char);
unsigned short stdc_bit_floor_us(unsigned short);
unsigned int stdc_bit_floor_ui(unsigned int);
unsigned long stdc_bit_floor_ul(unsigned long);
unsigned long long stdc_bit_floor_ull(unsigned long long);

unsigned char stdc_bit_ceil_uc(unsigned char);
unsigned short stdc_bit_ceil_us(unsigned short);
unsigned int stdc_bit_ceil_ui(unsigned int);
unsigned long stdc_bit_ceil_ul(unsigned long);
unsigned long long stdc_bit_ceil_ull(unsigned long long);

unsigned char stdc_rotate_left_uc(unsigned char, unsigned int);
unsigned short stdc_rotate_left_us(unsigned short, unsigned int);
unsigned int stdc_rotate_left_ui(unsigned int, unsigned int);
unsigned long stdc_rotate_left_ul(unsigned long, unsigned int);
unsigned long long stdc_rotate_left_ull(unsigned long long, unsigned int);

unsigned char stdc_rotate_right_uc(unsigned char, unsigned int);
unsigned short stdc_rotate_right_us(unsigned short, unsigned int);
unsigned int stdc_rotate_right_ui(unsigned int, unsigned int);
unsigned long stdc_rotate_right_ul(unsigned long, unsigned int);
unsigned long long stdc_rotate_right_ull(unsigned long long, unsigned int);

#endif
