include(FindLibcCommonUtils)
if(NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(UBSAN_OFFLOAD_SOURCES
  ubsan_offload_hsa.cpp
  ubsan_offload_hsa_interceptors.cpp
  ubsan_offload_rpc.cpp
  ubsan_offload_report.cpp
  ubsan_offload_symbolize.cpp
  )

set(UBSAN_OFFLOAD_HEADERS
  hsa.h
  ubsan_offload.h
  ubsan_offload_hsa.h
  ubsan_offload_rpc.h
  ubsan_offload_symbolize.h
  ../ubsan_offload_packet.h
  ../ubsan_offload_checks.inc
  ../ubsan_handlers_internal.h
  )

# No OBJECT_LIBS: sanitizer_common already comes from ubsan_standalone.
add_compiler_rt_runtime(clang_rt.ubsan_offload
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES ${UBSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${UBSAN_OFFLOAD_HEADERS}
  CFLAGS ${UBSAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET ubsan)
