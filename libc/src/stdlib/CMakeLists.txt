if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_header_library(
  abs_utils
  HDRS
    abs_utils.h
)

add_entrypoint_object(
  _Exit
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}._Exit
)

add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    abort.h
  DEPENDS
    libc.include.stdlib
    libc.src.signal.raise
    ._Exit
)

add_entrypoint_object(
  abs
  SRCS
    abs.cpp
  HDRS
    abs.h
  DEPENDS
    .abs_utils
)

add_entrypoint_object(
  labs
  SRCS
    labs.cpp
  HDRS
    labs.h
  DEPENDS
    .abs_utils
)

add_entrypoint_object(
  llabs
  SRCS
    llabs.cpp
  HDRS
    llabs.h
  DEPENDS
    .abs_utils
)
