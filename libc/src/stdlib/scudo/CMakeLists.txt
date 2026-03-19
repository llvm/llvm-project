set(SCUDO_STANDALONE_SOURCE_DIR
    ${LIBC_SOURCE_DIR}/../compiler-rt/lib/scudo/standalone)

set(SCUDO_INTERNAL_COMPILE_OPTIONS
  -DSCUDO_USE_LLVM_LIBC_INTERNAL_HEADERS
  -ffreestanding
  -fno-exceptions
  -Wno-deprecated-copy-with-user-provided-dtor
  -Wno-global-constructors
  -Wno-pedantic
  -Wno-unused-parameter
  -I${LIBC_SOURCE_DIR}/../compiler-rt/lib
  -I${SCUDO_STANDALONE_SOURCE_DIR}
  -I${SCUDO_STANDALONE_SOURCE_DIR}/include
)

set(SCUDO_CORE_SOURCES
  ${SCUDO_STANDALONE_SOURCE_DIR}/checksum.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/common.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/condition_variable_linux.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/crc32_hw.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/flags_parser.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/flags.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/linux.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/mem_map.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/mem_map_linux.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/release.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/report.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/report_linux.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/string_utils.cpp
  ${SCUDO_STANDALONE_SOURCE_DIR}/timing.cpp
)

set(SCUDO_CORE_DEPS
  libc.src.__support.StringUtil.error_to_string
  libc.src.__support.OSUtil.linux.auxv
  libc.src.__support.OSUtil.linux.syscall_wrappers.close
  libc.src.__support.OSUtil.linux.syscall_wrappers.open
  libc.src.__support.OSUtil.linux.syscall_wrappers.read
  libc.src.__support.OSUtil.linux.syscall_wrappers.write
  libc.src.__support.OSUtil.osutil
  libc.src.__support.libc_errno
  libc.src.__support.str_to_integer
  libc.src.__support.threads.identifier
  libc.src.__support.time.clock_gettime
)

if(TARGET libc.src.__support.threads.thread)
  list(APPEND SCUDO_CORE_DEPS libc.src.__support.threads.thread)
endif()
if(TARGET libc.src.__support.threads.callonce)
  list(APPEND SCUDO_CORE_DEPS libc.src.__support.threads.callonce)
endif()

add_object_library(
  core
  SRCS
    ${SCUDO_CORE_SOURCES}
  HDRS
    scudo_allocator.h
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    ${SCUDO_CORE_DEPS}
)

set(SCUDO_ALLOCATOR_DEPS
  .core
)
if(TARGET libc.src.__support.threads.fork_callbacks)
  list(APPEND SCUDO_ALLOCATOR_DEPS libc.src.__support.threads.fork_callbacks)
endif()

add_object_library(
  allocator
  SRCS
    allocator.cpp
  HDRS
    scudo_allocator.h
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    ${SCUDO_ALLOCATOR_DEPS}
)

add_entrypoint_object(
  malloc
  SRCS
    malloc.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  calloc
  SRCS
    calloc.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  realloc
  SRCS
    realloc.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  aligned_alloc
  SRCS
    aligned_alloc.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  free
  SRCS
    free.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  mallopt
  SRCS
    mallopt.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)

add_entrypoint_object(
  posix_memalign
  SRCS
    posix_memalign.cpp
  COMPILE_OPTIONS
    ${SCUDO_INTERNAL_COMPILE_OPTIONS}
  DEPENDS
    .allocator
)
