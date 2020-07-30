add_libc_fuzzer(
  strcpy_fuzz
  SRCS
    strcpy_fuzz.cpp
  DEPENDS
    libc.src.string.memcpy
    libc.src.string.strcpy
    libc.src.string.strlen
)

add_libc_fuzzer(
  strcmp_fuzz
  SRCS
    strcmp_fuzz.cpp
  DEPENDS
    libc.src.string.strcmp
)
