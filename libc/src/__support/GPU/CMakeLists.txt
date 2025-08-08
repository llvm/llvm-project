# These utilities are GPU only.
if(NOT LIBC_TARGET_OS_IS_GPU)
  return()
endif()

if(LIBC_CONF_MALLOC_DISABLE_SLAB_RECLAIMING)
  list(APPEND malloc_config_copts "-DLIBC_CONF_MALLOC_DISABLE_SLAB_RECLAIMING")
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
    libc.hdr.stdint_proxy
    libc.src.__support.common
    libc.src.__support.RPC.rpc_client
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.new
    .utils
  ${malloc_config_copts}
)
