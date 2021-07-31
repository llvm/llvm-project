if (LLVM_LIBC_FULL_BUILD)
add_entrypoint_object(
  __errno_location
  SRCS
    __errno_location.cpp
  HDRS
    __errno_location.h
    llvmlibc_errno.h
)
else()
add_entrypoint_object(
  __errno_location
  SRCS
    dummy_errno.cpp
  HDRS
    dummy_errno.h
)
endif()
