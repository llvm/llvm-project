add_subdirectory(linux)

add_header_library(
  osutil
  HDRS
    io.h
    quick_exit.h
    syscall.h
  DEPENDS
    libc.src.__support.OSUtil.linux.linux_util
)
