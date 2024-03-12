add_entrypoint_object(
  malloc
  SRCS
    malloc.cpp
  HDRS
    ../malloc.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.GPU.allocator
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  HDRS
    ../free.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.RPC.rpc_client
)

add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    ../abort.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.GPU.allocator
)

add_entrypoint_object(
  atexit
  SRCS
    atexit.cpp
  HDRS
    ../atexit.h
  CXX_STANDARD
    20 # For constinit of the atexit callback list.
  DEPENDS
    libc.include.stdlib
    libc.src.__support.fixedstack
)
