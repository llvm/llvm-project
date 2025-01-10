add_startup_object(
  tls
  SRC
    tls.cpp
  DEPENDS
    libc.config.app_h
    libc.include.sys_mman
    libc.include.sys_syscall
    libc.src.__support.OSUtil.osutil
    libc.src.string.memory_utils.inline_memcpy
  COMPILE_OPTIONS
    -fno-omit-frame-pointer
    -ffreestanding # To avoid compiler warnings about calling the main function.
)

add_startup_object(
  start
  SRC
    start.cpp
  DEPENDS
    libc.config.app_h
  COMPILE_OPTIONS
    -fno-omit-frame-pointer
    -ffreestanding # To avoid compiler warnings about calling the main function.
)
