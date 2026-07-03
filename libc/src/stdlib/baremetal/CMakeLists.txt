add_header_library(
  abort_utils
  HDRS
    abort_utils.h
)

add_entrypoint_object(
  malloc
  SRCS
    malloc.cpp
  HDRS
    ../malloc.h
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  HDRS
    ../free.h
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  calloc
  SRCS
    calloc.cpp
  HDRS
    ../calloc.h
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  realloc
  SRCS
    realloc.cpp
  HDRS
    ../realloc.h
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  aligned_alloc
  SRCS
    aligned_alloc.cpp
  HDRS
    ../aligned_alloc.h
  DEPENDS
    libc.src.__support.freelist_heap
)
