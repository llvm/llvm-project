add_custom_target(libc-stdbit-tests)

add_libc_test(
  stdc_leading_zeros_uc_test
  SUITE
    libc-stdbit-tests
  SRCS
    stdc_leading_zeros_uc_test.cpp
  DEPENDS
    libc.src.stdbit.stdc_leading_zeros_uc
)

add_libc_test(
  stdc_leading_zeros_us_test
  SUITE
    libc-stdbit-tests
  SRCS
    stdc_leading_zeros_us_test.cpp
  DEPENDS
    libc.src.stdbit.stdc_leading_zeros_us
)

add_libc_test(
  stdc_leading_zeros_ui_test
  SUITE
    libc-stdbit-tests
  SRCS
    stdc_leading_zeros_ui_test.cpp
  DEPENDS
    libc.src.stdbit.stdc_leading_zeros_ui
)

add_libc_test(
  stdc_leading_zeros_ul_test
  SUITE
    libc-stdbit-tests
  SRCS
    stdc_leading_zeros_ul_test.cpp
  DEPENDS
    libc.src.stdbit.stdc_leading_zeros_ul
)

add_libc_test(
  stdc_leading_zeros_ull_test
  SUITE
    libc-stdbit-tests
  SRCS
    stdc_leading_zeros_ull_test.cpp
  DEPENDS
    libc.src.stdbit.stdc_leading_zeros_ull
)
