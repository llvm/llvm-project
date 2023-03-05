add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  HDRS
    libc_errno.h     # Include this
    llvmlibc_errno.h # DEPRECATED: Will be removed soon
  DEPENDS
    libc.include.errno
)
