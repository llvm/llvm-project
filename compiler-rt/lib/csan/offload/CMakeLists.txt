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

add_compiler_rt_object_libraries(RTCsan_offload
  ARCHS ${CSAN_SUPPORTED_ARCH}
  SOURCES ${CSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${CSAN_OFFLOAD_HEADERS}
  CFLAGS ${CSAN_CFLAGS})
foreach(arch ${CSAN_SUPPORTED_ARCH})
  target_link_libraries(RTCsan_offload.${arch} PRIVATE
                        llvm-libc-common-utilities)
endforeach()

# The shared runtime embeds these objects directly, the static archive is only
# for static runtimes and is pulled in with -u.
add_compiler_rt_runtime(clang_rt.csan_offload
  STATIC
  ARCHS ${CSAN_SUPPORTED_ARCH}
  OBJECT_LIBS RTCsan_offload
              RTSanitizerOffload
  CFLAGS ${CSAN_CFLAGS}
  PARENT_TARGET csan)

add_compiler_rt_runtime(clang_rt.csan_offload-preinit
  STATIC
  ARCHS ${CSAN_SUPPORTED_ARCH}
  SOURCES csan_offload_preinit.cpp
  ADDITIONAL_HEADERS ${CSAN_OFFLOAD_HEADERS}
  CFLAGS ${CSAN_CFLAGS}
  PARENT_TARGET csan)
