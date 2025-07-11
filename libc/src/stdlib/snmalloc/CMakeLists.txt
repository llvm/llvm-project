set(SNMALLOC_USE_SELF_VENDORED_STL ON CACHE BOOL "use freestanding snmalloc setup" FORCE)
set(SNMALLOC_BUILD_TESTING OFF CACHE BOOL "disable snmalloc tests" FORCE)
set(SNMALLOC_HEADER_ONLY_LIBRARY ON CACHE BOOL "use snmalloc as header only library" FORCE)

set(CMAKE_DISABLE_FIND_PACKAGE_Backtrace ON)

add_subdirectory(${LIBC_INCLUDE_SNMALLOC} ${CMAKE_CURRENT_BINARY_DIR}/snmalloc EXCLUDE_FROM_ALL)

target_compile_options(
  snmalloc
  INTERFACE
    -ffreestanding
    -nostdinc
    -Wno-newline-eof
    -Wno-extra-semi
    -Wno-unused-command-line-argument
    -Wno-ctad-maybe-unsupported
    -isystem ${COMPILER_RESOURCE_DIR}/include
    -isystem ${LIBC_INCLUDE_DIR}
)

add_dependencies(snmalloc libc-headers)

set(SNMALLOC_DEPS
  # includes
  libc.include.errno
  libc.include.fcntl
  libc.include.limits
  libc.include.pthread
  libc.include.stdio
  libc.include.stdint
  libc.include.stdlib
  libc.include.string
  libc.include.strings
  libc.include.sys_mman
  libc.include.sys_prctl
  libc.include.sys_random
  libc.include.sys_syscall
  libc.include.sys_uio
  libc.include.unistd
  # symbols
  libc.src.errno.errno
  libc.src.fcntl.open
  libc.src.pthread.pthread_atfork
  libc.src.pthread.pthread_setspecific
  libc.src.pthread.pthread_key_create
  libc.src.stdlib.abort
  libc.src.string.memset
  libc.src.string.strlen
  libc.src.sys.mman.madvise
  libc.src.sys.mman.mmap
  libc.src.sys.uio.writev
  libc.src.time.clock_gettime
  libc.src.unistd.__llvm_libc_syscall
  libc.src.unistd.close
  libc.src.unistd.fsync
  libc.src.unistd.getentropy
  libc.src.unistd.read
)

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
    ${SNMALLOC_DEPS}
)

add_entrypoint_object(
  realloc
  SRCS
    realloc.cpp
  DEPENDS
    snmalloc
    ${SNMALLOC_DEPS}
)

add_entrypoint_object(
  aligned_alloc
  SRCS
    aligned_alloc.cpp
  DEPENDS
    snmalloc
    ${SNMALLOC_DEPS}
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  DEPENDS
    snmalloc
    ${SNMALLOC_DEPS}
)

add_entrypoint_object(
  mallopt
  SRCS
    mallopt.cpp
  DEPENDS
    snmalloc
    ${SNMALLOC_DEPS}
)
