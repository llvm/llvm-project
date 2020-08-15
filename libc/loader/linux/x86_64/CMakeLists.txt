add_loader_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    libc.config.linux.linux_syscall_h
    libc.config.linux.app_h
    libc.include.sys_syscall
    libc.src.string.memcpy
    libc.src.sys.mman.mmap
  COMPILE_OPTIONS
    -fno-omit-frame-pointer
    -ffreestanding # To avoid compiler warnings about calling the main function.
)
