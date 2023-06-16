if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  return()
endif()

add_header_library(
  rpc
  HDRS
    rpc.h
    rpc_util.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.functional
    libc.src.__support.GPU.utils
)

add_object_library(
  rpc_client
  SRCS
    rpc_client.cpp
  HDRS
    rpc_client.h
  DEPENDS
    libc.include.gpu_rpc
    libc.src.__support.GPU.utils
    .rpc
)
