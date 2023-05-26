add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  HDRS
    libc_errno.h     # Include this
  DEPENDS
    libc.include.errno
    libc.src.__support.common
)
