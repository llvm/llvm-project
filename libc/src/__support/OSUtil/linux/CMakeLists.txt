if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})

add_object_library(
  linux_util
  SRCS
    exit.cpp
    fcntl.cpp
  HDRS
    io.h
    syscall.h
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.linux_${LIBC_TARGET_ARCHITECTURE}_util
    libc.src.__support.common
    libc.src.__support.CPP.string_view
    libc.src.errno.errno
    libc.hdr.fcntl_macros
    libc.hdr.types.struct_flock
    libc.hdr.types.struct_flock64
    libc.hdr.types.struct_f_owner_ex
    libc.hdr.types.off_t
)

add_header_library(
  vdso_sym
  HDRS
    vdso_sym.h
  DEPENDS
    libc.src.__support.common
)

add_object_library(
  vdso
  HDRS
    vdso.h
  SRCS
    vdso.cpp
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.vdso
    libc.src.__support.CPP.array
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.string_view
    libc.src.__support.threads.callonce
    libc.src.__support.threads.linux.futex_word_type
    libc.hdr.types.struct_timeval
    libc.hdr.types.struct_timespec
    libc.hdr.types.clockid_t
    libc.hdr.types.time_t
    libc.hdr.link_macros
    libc.src.errno.errno
    libc.src.sys.auxv.getauxval
)

add_object_library(
    random
  HDRS
    random.h
  SRCS
    random.cpp
  DEPENDS
    libc.src.sys.random.getrandom
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap
    libc.src.unistd.sysconf
    libc.src.errno.errno
    libc.src.__support.common
    libc.src.__support.OSUtil.linux.vdso
    libc.src.__support.threads.callonce
    libc.src.__support.threads.linux.raw_mutex
    libc.src.__support.threads.thread
    libc.src.sched.sched_getaffinity
    libc.src.sched.__sched_getcpucount
)

