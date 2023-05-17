add_header_library(
  linux_arm_util
  HDRS
    syscall.h
  DEPENDS
    libc.src.__support.common
)
