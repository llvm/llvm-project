add_object_library(
  rand_util
  SRCS
    rand_util.cpp
  HDRS
    rand_util.h
  DEPENDS
    libc.src.__support.common
)

add_entrypoint_object(
  rand
  SRCS
    rand.cpp
  HDRS
    ../rand.h
  DEPENDS
    .rand_util
    libc.include.stdlib
)

add_entrypoint_object(
  srand
  SRCS
    srand.cpp
  HDRS
    srand.h
  DEPENDS
    .rand_util
    libc.include.stdlib
)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    ../abort.h
)
