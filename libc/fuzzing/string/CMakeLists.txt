add_libc_fuzzer(
  strcmp_fuzz
  SRCS
    strcmp_fuzz.cpp
  DEPENDS
    libc.src.string.strcmp
)

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
  strstr_fuzz
  SRCS
    strstr_fuzz.cpp
  DEPENDS
    libc.src.string.strstr
    libc.src.string.strlen
)

add_libc_fuzzer(
  memcpy_fuzz
  SRCS
    memcpy_fuzz.cpp
  DEPENDS
    libc.src.string.memcpy
)

add_libc_fuzzer(
  memset_fuzz
  SRCS
    memset_fuzz.cpp
  DEPENDS
    libc.src.string.memset
)

add_libc_fuzzer(
  memcmp_fuzz
  SRCS
    memcmp_fuzz.cpp
  DEPENDS
    libc.src.string.memcmp
)

add_libc_fuzzer(
  bcmp_fuzz
  SRCS
    bcmp_fuzz.cpp
  DEPENDS
    libc.src.string.bcmp
)
