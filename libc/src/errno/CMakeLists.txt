# If we are in full build mode, we will provide the errno definition ourselves,
# and if we are in overlay mode, we will just re-use the system's errno.
# We are passing LIBC_FULL_BUILD flag in full build mode so that the
# implementation of libc_errno will know if we are in full build mode or not.

add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.src.__support.common
)
