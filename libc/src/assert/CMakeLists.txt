add_entrypoint_object(
  __assert_fail
  SRCS
    __assert_fail.cpp
  HDRS
    assert.h
  DEPENDS
    abort
    # These two dependencies are temporary and should be replaced by fprintf
    # later.
    sys_syscall_h
    linux_syscall_h
)
