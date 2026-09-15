if(NOT LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_object_library(
  rpc_client
  SRCS
    rpc_client.cpp
    emissary_device_utils.cpp
  HDRS
    rpc_client.h
  DEPENDS
    libc.src.__support.GPU.utils
)
