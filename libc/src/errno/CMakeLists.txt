if (LLVM_LIBC_FULL_BUILD)
add_object_library(
  errno
  SRCS
    errno.cpp
  HDRS
    llvmlibc_errno.h
)
else()
add_object_library(
  errno
  SRCS
    dummy_errno.cpp
  HDRS
    dummy_errno.h
)
endif()
