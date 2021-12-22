add_entrypoint_object(
  __assert_fail
  SRCS
    __assert_fail.cpp
  HDRS
    __assert_fail.h
    assert.h
  DEPENDS
    # These two dependencies are temporary and should be replaced by fprintf
    # later.
    libc.src.__support.OSUtil.osutil
    libc.include.sys_syscall
    libc.src.stdlib.abort
)
