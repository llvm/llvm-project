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

#endif
