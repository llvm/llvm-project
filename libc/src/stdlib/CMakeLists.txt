add_entrypoint_object(
  atoi
  SRCS
    atoi.cpp
  HDRS
    atoi.h
  DEPENDS
    libc.src.__support.str_conv_utils
)

add_entrypoint_object(
  atol
  SRCS
    atol.cpp
  HDRS
    atol.h
  DEPENDS
    libc.src.__support.str_conv_utils
)

add_entrypoint_object(
  atoll
  SRCS
    atoll.cpp
  HDRS
    atoll.h
  DEPENDS
    libc.src.__support.str_conv_utils
)

add_entrypoint_object(
  strtol
  SRCS
    strtol.cpp
  HDRS
    strtol.h
  DEPENDS
    libc.src.__support.str_conv_utils
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

add_entrypoint_object(
  strtoul
  SRCS
    strtoul.cpp
  HDRS
    strtoul.h
  DEPENDS
    libc.src.__support.str_conv_utils
)

add_entrypoint_object(
  strtoull
  SRCS
    strtoull.cpp
  HDRS
    strtoull.h
  DEPENDS
    libc.src.__support.str_conv_utils
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
  div
  SRCS
    div.cpp
  HDRS
    div.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  ldiv
  SRCS
    ldiv.cpp
  HDRS
    ldiv.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  lldiv
  SRCS
    lldiv.cpp
  HDRS
    lldiv.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  bsearch
  SRCS
    bsearch.cpp
  HDRS
    bsearch.h
  DEPENDS
    libc.include.stdlib
)

add_entrypoint_object(
  qsort
  SRCS
    qsort.cpp
  HDRS
    qsort.h
  DEPENDS
    libc.include.stdlib
)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

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
