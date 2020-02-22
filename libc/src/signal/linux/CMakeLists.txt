
add_entrypoint_object(
  raise
  SRCS
    raise.cpp
  HDRS
    signal.h
    ../raise.h
  DEPENDS
    sys_syscall_h
    linux_syscall_h
    __errno_location
    signal_h
)
