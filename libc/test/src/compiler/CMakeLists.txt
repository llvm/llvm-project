add_custom_target(libc_stack_chk_guard_unittests)

add_libc_unittest(
  stack_chk_guard_test
  SUITE
    libc_stack_chk_guard_unittests
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
