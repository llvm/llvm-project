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

# Since this test depends on a macro defined in the sched header, it won't work
# without fullbuild.
if(LLVM_LIBC_FULL_BUILD)
  add_libc_unittest(
    cpu_count_test
    SUITE
      libc_sched_unittests
    SRCS
      cpu_count_test.cpp
    DEPENDS
      libc.include.errno
      libc.include.sched
      libc.include.sys_syscall
      libc.src.__support.OSUtil.osutil
      libc.src.sched.sched_getaffinity
      libc.src.sched.sched_getcpucount
      libc.test.errno_setter_matcher
  )
endif()
