set(SNMALLOC_USE_SELF_VENDORED_STL ON CACHE BOOL "use freestanding snmalloc setup" FORCE)
set(SNMALLOC_BUILD_TESTING OFF CACHE BOOL "disable snmalloc tests" FORCE)
set(SNMALLOC_HEADER_ONLY_LIBRARY ON CACHE BOOL "use snmalloc as header only library" FORCE)

# Disable installation
macro (install)
endmacro ()

add_subdirectory(${LLVM_LIBC_INCLUDE_SNMALLOC} ${CMAKE_CURRENT_BINARY_DIR}/snmalloc EXCLUDE_FROM_ALL)

target_compile_options(
  snmalloc 
  INTERFACE 
    -ffreestanding
    -nostdinc 
    -Wno-newline-eof 
    -Wno-extra-semi
    -Wno-unused-command-line-argument
    -Wno-ctad-maybe-unsupported
    # TODO: define this
    -DSTDERR_FILENO=2
    -DSNMALLOC_USE_PTHREAD_DESTRUCTORS
    # include_directories does not propagate, use options instead
    -isystem ${COMPILER_RESOURCE_DIR}/include
    -isystem ${LIBC_INCLUDE_DIR}
)
add_dependencies(snmalloc libc-headers)

add_entrypoint_object(
  malloc
  SRCS
    malloc.cpp
  DEPENDS
    snmalloc
)

add_entrypoint_object(
  calloc
  SRCS
    calloc.cpp
  DEPENDS
    snmalloc
)

add_entrypoint_object(
  realloc
  SRCS
    realloc.cpp
  DEPENDS
    snmalloc
)

add_entrypoint_object(
  aligned_alloc
  SRCS
    aligned_alloc.cpp
  DEPENDS
    snmalloc
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  DEPENDS
    snmalloc
)

add_entrypoint_object(
  mallopt
  SRCS
    mallopt.cpp
  DEPENDS
    snmalloc
)
