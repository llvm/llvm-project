if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})

add_header_library(
  linux_util
  HDRS
    io.h
    quick_exit.h
    syscall.h
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.linux_${LIBC_TARGET_ARCHITECTURE}_util
    libc.src.__support.common
    libc.src.__support.CPP.string_view
)
