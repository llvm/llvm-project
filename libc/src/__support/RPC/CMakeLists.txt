add_header_library(
  rpc
  HDRS
    rpc.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.functional
)

if(NOT LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_object_library(
  rpc_client
  SRCS
    rpc_client.cpp
  HDRS
    rpc_client.h
  DEPENDS
    libc.src.__support.GPU.utils
)
