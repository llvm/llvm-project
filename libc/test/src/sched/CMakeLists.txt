add_libc_testsuite(libc_sched_unittests)

add_libc_unittest(
  affinity_test
  SUITE
    libc_sched_unittests
  SRCS
    affinity_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.sched
    libc.include.sys_syscall
    libc.src.__support.OSUtil.osutil
    libc.src.sched.sched_getaffinity
    libc.src.sched.sched_setaffinity
    libc.test.errno_setter_matcher
)
