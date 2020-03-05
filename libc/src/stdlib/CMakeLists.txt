if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    abort.h
  DEPENDS
    raise
    _Exit
    stdlib_h
)
