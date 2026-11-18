add_libc_testsuite(libc-stack-chk-guard-tests)

if(LIBC_TARGET_OS_IS_BAREMETAL)
  return()
endif()

add_libc_test(
  stack_chk_guard_test
  SUITE
    libc-stack-chk-guard-tests
  SRCS
    stack_chk_guard_test.cpp
  DEPENDS
    libc.hdr.signal_macros
    libc.src.__support.macros.sanitizer
    libc.src.compiler.__stack_chk_fail
    libc.src.string.memset
  COMPILE_OPTIONS
    -fstack-protector-all
)
