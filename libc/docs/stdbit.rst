=========================
Bitwise Utility Functions
=========================

.. include:: check.rst

---------------
Source location
---------------

-   The main source for bitwise utility functions is located at:
    ``libc/src/stdbit``.

-   The source for internal helpers used to implement these is located at:
    ``libc/src/__support/CPP/bit.h``.

-   The tests are located at:
    ``libc/test/src/stdbit/``, ``libc/test/include/stdbit_test.cpp``, and
    ``src/__support/CPP/bit_test.cpp``.

---------------------
Implementation Status
---------------------

Functions
=========

..
  Do not order these, they are as they appear in the standard.

============================ =========
Function Name                Available
============================ =========
stdc_leading_zeros_uc        |check|
stdc_leading_zeros_us        |check|
stdc_leading_zeros_ui        |check|
stdc_leading_zeros_ul        |check|
stdc_leading_zeros_ull       |check|
stdc_leading_ones_uc
stdc_leading_ones_us
stdc_leading_ones_ui
stdc_leading_ones_ul
stdc_leading_ones_ull
stdc_trailing_zeros_uc
stdc_trailing_zeros_us
stdc_trailing_zeros_ui
stdc_trailing_zeros_ul
stdc_trailing_zeros_ull
stdc_trailing_ones_uc
stdc_trailing_ones_us
stdc_trailing_ones_ui
stdc_trailing_ones_ul
stdc_trailing_ones_ull
stdc_first_leading_zero_uc
stdc_first_leading_zero_us
stdc_first_leading_zero_ui
stdc_first_leading_zero_ul
stdc_first_leading_zero_ull
stdc_first_leading_one_uc
stdc_first_leading_one_us
stdc_first_leading_one_ui
stdc_first_leading_one_ul
stdc_first_leading_one_ull
stdc_first_trailing_zero_uc
stdc_first_trailing_zero_us
stdc_first_trailing_zero_ui
stdc_first_trailing_zero_ul
stdc_first_trailing_zero_ull
stdc_first_trailing_one_uc
stdc_first_trailing_one_us
stdc_first_trailing_one_ui
stdc_first_trailing_one_ul
stdc_first_trailing_one_ull
stdc_count_zeros_uc
stdc_count_zeros_us
stdc_count_zeros_ui
stdc_count_zeros_ul
stdc_count_zeros_ull
stdc_count_ones_uc
stdc_count_ones_us
stdc_count_ones_ui
stdc_count_ones_ul
stdc_count_ones_ull
stdc_has_single_bit_uc
stdc_has_single_bit_us
stdc_has_single_bit_ui
stdc_has_single_bit_ul
stdc_has_single_bit_ull
stdc_bit_width_uc
stdc_bit_width_us
stdc_bit_width_ui
stdc_bit_width_ul
stdc_bit_width_ull
stdc_bit_floor_uc
stdc_bit_floor_us
stdc_bit_floor_ui
stdc_bit_floor_ul
stdc_bit_floor_ull
stdc_bit_ceil_uc
stdc_bit_ceil_us
stdc_bit_ceil_ui
stdc_bit_ceil_ul
stdc_bit_ceil_ull
============================ =========


Macros
======

=========================  =========
Macro Name                 Available
=========================  =========
__STDC_VERSION_STDBIT_H__
__STDC_ENDIAN_LITTLE__
__STDC_ENDIAN_BIG__
__STDC_ENDIAN_NATIVE__
stdc_leading_zeros         |check|
stdc_leading_ones
stdc_trailing_zeros
stdc_trailing_ones
stdc_first_leading_zero
stdc_first_leading_one
stdc_first_trailing_zero
stdc_first_trailing_one
stdc_count_zeros
stdc_count_ones
stdc_has_single_bit
stdc_bit_width
stdc_bit_floor
stdc_bit_ceil
=========================  =========

Standards
=========
stdbit.h was specified as part of C23 in section 7.18 "Bit and byte utilities."
