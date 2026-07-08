set(freelist_entrypoint_compile_options "")
if(LIBC_CONF_FREELISTHEAP_PRECISE_ALLOC_SIZE)
  list(APPEND freelist_entrypoint_compile_options "-DLIBC_COPT_FREELISTHEAP_PRECISE_ALLOC_SIZE")
endif()

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
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  HDRS
    ../free.h
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  calloc
  SRCS
    calloc.cpp
  HDRS
    ../calloc.h
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  realloc
  SRCS
    realloc.cpp
  HDRS
    ../realloc.h
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  aligned_alloc
  SRCS
    aligned_alloc.cpp
  HDRS
    ../aligned_alloc.h
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)

add_entrypoint_object(
  malloc_usable_size
  SRCS
    malloc_usable_size.cpp
  HDRS
    ../malloc_usable_size.h
  COMPILE_OPTIONS
    ${freelist_entrypoint_compile_options}
  DEPENDS
    libc.src.__support.freelist_heap
)
