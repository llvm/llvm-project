add_loader_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    libc.config.linux.linux_syscall_h
    libc.include.sys_syscall
  COMPILE_OPTIONS
    -fno-omit-frame-pointer
    -ffreestanding # To avoid compiler warnings about calling the main function.
)
