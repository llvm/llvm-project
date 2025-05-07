# If we are in full build mode, we will provide the errno definition ourselves,
# and if we are in overlay mode, we will just re-use the system's errno.
# We are passing LIBC_FULL_BUILD flag in full build mode so that the
# implementation of libc_errno will know if we are in full build mode or not.

# TODO: Move LIBC_FULL_BUILD flag to _get_common_compile_options.
set(full_build_flag "")
if(LLVM_LIBC_FULL_BUILD)
  set(full_build_flag "-DLIBC_FULL_BUILD")
endif()

if(LIBC_CONF_ERRNO_MODE)
  set(errno_config_copts "-DLIBC_ERRNO_MODE=${LIBC_CONF_ERRNO_MODE}")
endif()

add_entrypoint_object(
  errno
  SRCS
    libc_errno.cpp
  HDRS
    libc_errno.h     # Include this
  COMPILE_OPTIONS
    ${full_build_flag}
    ${errno_config_copts}
  DEPENDS
    libc.hdr.errno_macros
    libc.src.__support.common
)
