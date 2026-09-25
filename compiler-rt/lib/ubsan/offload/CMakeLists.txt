include(FindLibcCommonUtils)
if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(UBSAN_OFFLOAD_SOURCES
  ubsan_offload_hsa_interceptors.cpp
  ubsan_offload_report.cpp
  )

set(UBSAN_OFFLOAD_HEADERS
  ubsan_offload.h
  ../ubsan_offload_packet.h
  ../ubsan_offload_checks.inc
  ../ubsan_handlers_internal.h
  )

add_compiler_rt_runtime(clang_rt.ubsan_offload
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES ${UBSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${UBSAN_OFFLOAD_HEADERS}
  OBJECT_LIBS RTSanitizerOffload
  CFLAGS ${UBSAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET ubsan)
