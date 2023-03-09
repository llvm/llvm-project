add_header_library(
  linux_riscv64_util
  HDRS
    syscall.h
  DEPENDS
    libc.src.__support.common
)
