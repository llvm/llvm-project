# These utilities are GPU only.
if(NOT LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_header_library(
  utils
  HDRS
    utils.h
)

add_object_library(
  allocator
  SRCS
    allocator.cpp
  HDRS
    allocator.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.RPC.rpc_client
    .utils
)
