add_header_library(
  linux_hexagon_util
  HDRS
    syscall.h
  DEPENDS
    libc.src.__support.common
)

