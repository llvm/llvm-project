# If we are in full build mode, we will provide the errno definition ourselves,
# and if we are in overlay mode, we will just re-use the system's errno.

add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  HDRS
    ../__support/libc_errno.h
  DEPENDS
    libc.hdr.errno_macros
    libc.src.__support.common
    libc.src.__support.libc_errno
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
)
