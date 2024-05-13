add_libc_fuzzer(
  printf_parser_fuzz
  SRCS
    printf_parser_fuzz.cpp
  DEPENDS
    libc.src.stdio.printf_core.parser
)

add_libc_fuzzer(
  printf_float_conv_fuzz
  NEED_MPFR
  SRCS
    printf_float_conv_fuzz.cpp
  DEPENDS
    libc.src.stdio.snprintf
    libc.src.__support.FPUtil.fp_bits
)

if(LIBC_COMPILER_HAS_FIXED_POINT)
  add_libc_fuzzer(
    printf_fixed_conv_fuzz
    NEED_MPFR
    SRCS
      printf_fixed_conv_fuzz.cpp
    DEPENDS
      libc.src.stdio.snprintf
      libc.src.__support.fixed_point.fx_bits
    COMPILE_OPTIONS
      -ffixed-point # TODO: add -ffixed-point to fuzz tests automatically
  )
endif()
