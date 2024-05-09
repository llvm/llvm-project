if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})

add_object_library(
  linux_util
  SRCS
    quick_exit.cpp
  HDRS
    io.h
    syscall.h
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.linux_${LIBC_TARGET_ARCHITECTURE}_util
    libc.src.__support.common
    libc.src.__support.CPP.string_view
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
    libc.src.__support.CPP.string_view
    libc.src.__support.threads.callonce
    libc.src.__support.threads.linux.futex_word_type
    libc.src.errno.errno
    libc.src.sys.auxv.getauxval
)
