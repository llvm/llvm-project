if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

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
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  labs
  SRCS
    labs.cpp
  HDRS
    labs.h
  DEPENDS
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  llabs
  SRCS
    llabs.cpp
  HDRS
    llabs.h
  DEPENDS
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  strtoll
  SRCS
    strtoll.cpp
  HDRS
    strtoll.h
  DEPENDS
    libc.src.__support.str_conv_utils
)
