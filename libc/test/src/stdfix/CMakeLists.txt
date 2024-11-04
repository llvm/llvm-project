if(NOT LIBC_COMPILER_HAS_FIXED_POINT)
  return()
endif()

add_custom_target(libc-stdfix-tests)

foreach(suffix IN ITEMS hr r lr hk k lk)
  add_libc_test(
    abs${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      AbsTest.h
    SRCS
      abs${suffix}_test.cpp
    COMPILE_OPTIONS
      -O3
      -ffixed-point
    DEPENDS
      libc.src.stdfix.abs${suffix}
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()

foreach(suffix IN ITEMS hr r lr hk k lk uhr ur ulr uhk uk ulk)
  add_libc_test(
    round${suffix}_test
    SUITE
      libc-stdfix-tests
    HDRS
      RoundTest.h
    SRCS
      round${suffix}_test.cpp
    COMPILE_OPTIONS
      -O3
      -ffixed-point
    DEPENDS
      libc.src.stdfix.round${suffix}
      libc.src.__support.fixed_point.fx_bits
  )
endforeach()
