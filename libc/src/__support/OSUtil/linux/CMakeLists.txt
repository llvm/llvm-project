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

add_object_library(
  pid
  SRCS
    pid.cpp
  HDRS
    ../pid.h
  DEPENDS
    libc.src.__support.OSUtil.osutil
    libc.src.__support.common
    libc.hdr.types.pid_t
    libc.include.sys_syscall
)
