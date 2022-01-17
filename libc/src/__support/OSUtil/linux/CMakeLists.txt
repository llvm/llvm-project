add_subdirectory(aarch64)
add_subdirectory(x86_64)

add_header_library(
  linux_util
  HDRS
    io.h
    quick_exit.h
    syscall.h
  DEPENDS
    .aarch64.linux_aarch64_util
    .x86_64.linux_x86_64_util
    libc.src.__support.common
)
