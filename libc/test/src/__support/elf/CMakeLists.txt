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

add_libc_test(
  relocation_test
  SUITE
    libc-support-tests
  SRCS
    relocation_test.cpp
  DEPENDS
    libc.src.__support.elf.relocation
)

add_libc_test(
  module_test
  SUITE
    libc-support-tests
  SRCS
    module_test.cpp
  DEPENDS
    libc.src.__support.elf.module
)

add_libc_test(
  tls_layout_test
  SUITE
    libc-support-tests
  SRCS
    tls_layout_test.cpp
  DEPENDS
    libc.src.__support.elf.tls_layout
)

add_libc_test(
  load_layout_test
  SUITE
    libc-support-tests
  SRCS
    load_layout_test.cpp
  DEPENDS
    libc.src.__support.elf.load_layout
)

add_libc_test(
  map_segments_test
  SUITE
    libc-support-tests
  SRCS
    map_segments_test.cpp
  DEPENDS
    libc.hdr.fcntl_macros
    libc.src.__support.OSUtil.linux.syscall_wrappers.close
    libc.src.__support.OSUtil.linux.syscall_wrappers.open
    libc.src.__support.OSUtil.linux.syscall_wrappers.write
    libc.src.__support.elf.map_segments
)
