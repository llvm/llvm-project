include(FindLibcCommonUtils)
if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(CSAN_OFFLOAD_SOURCES
  csan_offload_hsa_interceptors.cpp
  csan_offload_report.cpp
  )

set(CSAN_OFFLOAD_HEADERS
  csan_offload.h
  ../csan_defs.h
  ../csan_offload_packet.h
  )

add_compiler_rt_runtime(clang_rt.csan_offload
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES ${CSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${CSAN_OFFLOAD_HEADERS}
  OBJECT_LIBS RTSanitizerOffload
  CFLAGS ${CSAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET csan)
