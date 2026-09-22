# Build for the ConcurrencySanitizer runtime support library.

include_directories(.)
include_directories(..)
include_directories(../../include)

# GPU targets only need the watchpoint runtime. The host interceptor archive is
# built in the offload/ project.
if(COMPILER_RT_GPU_BUILD)
  include(FindLibcCommonUtils)
  if(NOT TARGET llvm-libc-common-utilities)
    add_compiler_rt_component(csan)
    return()
  endif()

  set(CSAN_SOURCES csan_gpu.cpp)
  set(CSAN_HEADERS csan_defs.h csan_offload_packet.h csan_watch.h)

  set(CSAN_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS)
  append_rtti_flag(OFF CSAN_CFLAGS)

  add_compiler_rt_component(csan)

  add_compiler_rt_runtime(clang_rt.csan
    STATIC
    ARCHS ${UBSAN_SUPPORTED_ARCH}
    SOURCES ${CSAN_SOURCES}
    ADDITIONAL_HEADERS ${CSAN_HEADERS}
    CFLAGS ${CSAN_CFLAGS}
    LINK_LIBS llvm-libc-common-utilities
    PARENT_TARGET csan)

  return()
endif()

set(CSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF CSAN_CFLAGS)

add_compiler_rt_component(csan)

add_compiler_rt_runtime(clang_rt.csan
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES csan.cpp csan_report.cpp
  ADDITIONAL_HEADERS csan.h csan_defs.h csan_flags.inc csan_watch.h
  OBJECT_LIBS RTSanitizerCommon
              RTSanitizerCommonLibc
              RTSanitizerCommonCoverage
              RTSanitizerCommonSymbolizer
              RTSanitizerCommonSymbolizerInternal
              RTInterception
  CFLAGS ${CSAN_CFLAGS}
  PARENT_TARGET csan)

# Thin host interceptors for offloading. Linked with -u to keep it separate.
if(OS_NAME MATCHES "Linux" AND NOT ANDROID)
  add_subdirectory(offload)
endif()
