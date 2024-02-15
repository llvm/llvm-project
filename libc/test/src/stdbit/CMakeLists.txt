add_custom_target(libc-stdbit-tests)

set(prefixes
  leading_zeros
  leading_ones
  trailing_zeros
  trailing_ones
  first_leading_zero
  first_leading_one
  first_trailing_zero
  first_trailing_one
)
set(suffixes c s i l ll)
foreach(prefix IN LISTS prefixes)
  foreach(suffix IN LISTS suffixes)
    add_libc_test(
      stdc_${prefix}_u${suffix}_test
      SUITE
        libc-stdbit-tests
      SRCS
        stdc_${prefix}_u${suffix}_test.cpp
      DEPENDS
        libc.src.__support.CPP.limits
        libc.src.stdbit.stdc_${prefix}_u${suffix}
    )
  endforeach()
endforeach()
