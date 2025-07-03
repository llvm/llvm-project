add_startup_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    libc.src.stdlib.atexit
    libc.src.stdlib.exit
    libc.src.string.memcpy
    libc.src.string.memset
    libc.startup.baremetal.init
    libc.startup.baremetal.fini
  COMPILE_OPTIONS
    -ffreestanding # To avoid compiler warnings about calling the main function.
    -fno-builtin
)
get_fq_target_name(crt1 fq_name)
