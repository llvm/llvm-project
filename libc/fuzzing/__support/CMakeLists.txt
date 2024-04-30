add_libc_fuzzer(
  uint_fuzz
  SRCS
    uint_fuzz.cpp
  DEPENDS
    libc.src.__support.big_int
)
