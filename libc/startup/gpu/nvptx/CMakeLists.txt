add_startup_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    libc.src.__support.RPC.rpc_client
    libc.src.__support.GPU.utils
    libc.src.stdlib.exit
    libc.src.stdlib.atexit
  COMPILE_OPTIONS
    -ffreestanding # To avoid compiler warnings about calling the main function.
    -fno-builtin
)
get_fq_target_name(crt1 fq_name)
