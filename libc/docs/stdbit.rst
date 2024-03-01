=========================
Bitwise Utility Functions
=========================

.. include:: check.rst

---------------
Source Location
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
stdc_leading_ones_uc         |check|
stdc_leading_ones_us         |check|
stdc_leading_ones_ui         |check|
stdc_leading_ones_ul         |check|
stdc_leading_ones_ull        |check|
stdc_trailing_zeros_uc       |check|
stdc_trailing_zeros_us       |check|
stdc_trailing_zeros_ui       |check|
stdc_trailing_zeros_ul       |check|
stdc_trailing_zeros_ull      |check|
stdc_trailing_ones_uc        |check|
stdc_trailing_ones_us        |check|
stdc_trailing_ones_ui        |check|
stdc_trailing_ones_ul        |check|
stdc_trailing_ones_ull       |check|
stdc_first_leading_zero_uc   |check|
stdc_first_leading_zero_us   |check|
stdc_first_leading_zero_ui   |check|
stdc_first_leading_zero_ul   |check|
stdc_first_leading_zero_ull  |check|
stdc_first_leading_one_uc    |check|
stdc_first_leading_one_us    |check|
stdc_first_leading_one_ui    |check|
stdc_first_leading_one_ul    |check|
stdc_first_leading_one_ull   |check|
stdc_first_trailing_zero_uc  |check|
stdc_first_trailing_zero_us  |check|
stdc_first_trailing_zero_ui  |check|
stdc_first_trailing_zero_ul  |check|
stdc_first_trailing_zero_ull |check|
stdc_first_trailing_one_uc   |check|
stdc_first_trailing_one_us   |check|
stdc_first_trailing_one_ui   |check|
stdc_first_trailing_one_ul   |check|
stdc_first_trailing_one_ull  |check|
stdc_count_zeros_uc          |check|
stdc_count_zeros_us          |check|
stdc_count_zeros_ui          |check|
stdc_count_zeros_ul          |check|
stdc_count_zeros_ull         |check|
stdc_count_ones_uc           |check|
stdc_count_ones_us           |check|
stdc_count_ones_ui           |check|
stdc_count_ones_ul           |check|
stdc_count_ones_ull          |check|
stdc_has_single_bit_uc       |check|
stdc_has_single_bit_us       |check|
stdc_has_single_bit_ui       |check|
stdc_has_single_bit_ul       |check|
stdc_has_single_bit_ull      |check|
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
stdc_leading_ones          |check|
stdc_trailing_zeros        |check|
stdc_trailing_ones         |check|
stdc_first_leading_zero    |check|
stdc_first_leading_one     |check|
stdc_first_trailing_zero   |check|
stdc_first_trailing_one    |check|
stdc_count_zeros           |check|
stdc_count_ones            |check|
stdc_has_single_bit        |check|
stdc_bit_width
stdc_bit_floor
stdc_bit_ceil
=========================  =========

Standards
=========
stdbit.h was specified as part of C23 in section 7.18 "Bit and byte utilities."
