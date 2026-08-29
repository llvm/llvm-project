include(FindLibcCommonUtils)
if(NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(UBSAN_DEVICE_SOURCES
  ubsan_device_hsa.cpp
  ubsan_device_hsa_interceptors.cpp
  ubsan_device_rpc.cpp
  ubsan_device_report.cpp
  ubsan_device_symbolize.cpp
  )

set(UBSAN_DEVICE_HEADERS
  hsa.h
  ubsan_device.h
  ubsan_device_hsa.h
  ubsan_device_rpc.h
  ubsan_device_symbolize.h
  ../ubsan_device_packet.h
  ../ubsan_device_checks.inc
  ../ubsan_handlers_internal.h
  )

# No OBJECT_LIBS: sanitizer_common already comes from ubsan_standalone.
add_compiler_rt_runtime(clang_rt.ubsan_device
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES ${UBSAN_DEVICE_SOURCES}
  ADDITIONAL_HEADERS ${UBSAN_DEVICE_HEADERS}
  CFLAGS ${UBSAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET ubsan)
