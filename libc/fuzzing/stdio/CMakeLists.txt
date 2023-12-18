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
