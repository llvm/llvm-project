include(FindLibcCommonUtils)
if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(ALL_UBSAN_OFFLOAD_SUPPORTED_ARCH ${X86_64} ${ARM64})
list_intersect(UBSAN_OFFLOAD_SUPPORTED_ARCH
  UBSAN_SUPPORTED_ARCH
  ALL_UBSAN_OFFLOAD_SUPPORTED_ARCH)

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

add_compiler_rt_object_libraries(RTUbsan_offload
  ARCHS ${UBSAN_OFFLOAD_SUPPORTED_ARCH}
  SOURCES ${UBSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${UBSAN_OFFLOAD_HEADERS}
  CFLAGS ${UBSAN_CFLAGS})
foreach(arch ${UBSAN_OFFLOAD_SUPPORTED_ARCH})
  target_link_libraries(RTUbsan_offload.${arch} PRIVATE
                        llvm-libc-common-utilities)
endforeach()

# The shared runtime embeds these objects directly, the static archive is only
# for static runtimes and is pulled in with -u.
add_compiler_rt_runtime(clang_rt.ubsan_offload
  STATIC
  ARCHS ${UBSAN_OFFLOAD_SUPPORTED_ARCH}
  OBJECT_LIBS RTUbsan_offload
              RTSanitizerOffload
  CFLAGS ${UBSAN_CFLAGS}
  PARENT_TARGET ubsan)

add_compiler_rt_runtime(clang_rt.ubsan_offload-preinit
  STATIC
  ARCHS ${UBSAN_OFFLOAD_SUPPORTED_ARCH}
  SOURCES ubsan_offload_preinit.cpp
  ADDITIONAL_HEADERS ${UBSAN_OFFLOAD_HEADERS}
  CFLAGS ${UBSAN_CFLAGS}
  PARENT_TARGET ubsan)
