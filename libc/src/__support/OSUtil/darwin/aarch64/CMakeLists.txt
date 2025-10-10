add_header_library(
  darwin_util
  HDRS
    syscall.h
  DEPENDS
    libc.src.__support.common
)
