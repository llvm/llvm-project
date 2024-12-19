add_custom_target(libc_sys_wait_unittests)

add_libc_unittest(
  waitpid_test
  SUITE
    libc_sys_stat_unittests
  SRCS
    waitpid_test.cpp
  DEPENDS
    libc.include.sys_wait
    libc.src.errno.errno 
    libc.src.sys.wait.waitpid
)

add_libc_unittest(
  wait4_test
  SUITE
    libc_sys_stat_unittests
  SRCS
    wait4_test.cpp
  DEPENDS
    libc.include.sys_wait
    libc.src.errno.errno 
    libc.src.sys.wait.wait4
)
