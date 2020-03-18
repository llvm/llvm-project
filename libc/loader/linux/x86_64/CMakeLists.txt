add_loader_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    linux_syscall_h
    sys_syscall_h
  COMPILE_OPTIONS
    -fno-omit-frame-pointer
    -ffreestanding # To avoid compiler warnings about calling the main function.
)
