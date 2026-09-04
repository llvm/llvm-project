add_libc_testsuite(libc-elf-tests)

add_libc_test(
  dynamic_test
  SUITE
    libc-support-tests
  SRCS
    dynamic_test.cpp
  DEPENDS
    libc.src.__support.elf.dynamic
)

add_libc_test(
  hash_test
  SUITE
    libc-support-tests
  SRCS
    hash_test.cpp
  DEPENDS
    libc.src.__support.elf.hash
)

add_libc_test(
  symbol_test
  SUITE
    libc-support-tests
  SRCS
    symbol_test.cpp
  DEPENDS
    libc.src.__support.elf.symbol
)
