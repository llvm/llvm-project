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
    ARCHS ${CSAN_SUPPORTED_ARCH}
    SOURCES ${CSAN_SOURCES}
    ADDITIONAL_HEADERS ${CSAN_HEADERS}
    CFLAGS ${CSAN_CFLAGS}
    LINK_LIBS llvm-libc-common-utilities
    PARENT_TARGET csan)

  return()
endif()

set(CSAN_SOURCES
  csan.cpp
  csan_report.cpp
  )

set(CSAN_HEADERS
  csan.h
  csan_defs.h
  csan_flags.inc
  csan_watch.h
  )

set(CSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF CSAN_CFLAGS)

set(CSAN_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

set(CSAN_DYNAMIC_LIBS
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_CXX_ABI_LIBRARIES}
  ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl CSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt CSAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread CSAN_DYNAMIC_LIBS)
if (COMPILER_RT_ENABLE_INTERNAL_SYMBOLIZER)
  append_list_if(COMPILER_RT_HAS_LIBM m CSAN_DYNAMIC_LIBS)
endif()

set(CSAN_COMMON_RUNTIME_OBJECT_LIBS
  RTSanitizerCommon
  RTSanitizerCommonLibc
  RTSanitizerCommonCoverage
  RTSanitizerCommonSymbolizer
  RTSanitizerCommonSymbolizerInternal
  RTInterception)

add_compiler_rt_component(csan)

add_compiler_rt_object_libraries(RTCsan
  ARCHS ${CSAN_SUPPORTED_ARCH}
  SOURCES ${CSAN_SOURCES}
  ADDITIONAL_HEADERS ${CSAN_HEADERS}
  CFLAGS ${CSAN_CFLAGS})

add_compiler_rt_runtime(clang_rt.csan
  STATIC
  ARCHS ${CSAN_SUPPORTED_ARCH}
  OBJECT_LIBS RTCsan
              ${CSAN_COMMON_RUNTIME_OBJECT_LIBS}
  CFLAGS ${CSAN_CFLAGS}
  PARENT_TARGET csan)

# Thin host interceptors for offloading. Linked with -u to keep it separate
# from static runtimes and embedded directly in the shared runtime.
if(OS_NAME MATCHES "Linux" AND NOT ANDROID)
  add_subdirectory(offload)
endif()

if(COMPILER_RT_HAS_VERSION_SCRIPT)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
  add_compiler_rt_object_libraries(RTCsan_dynamic_version_script_dummy
    ARCHS ${CSAN_SUPPORTED_ARCH}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
    CFLAGS ${CSAN_CFLAGS})

  foreach(arch ${CSAN_SUPPORTED_ARCH})
    set(CSAN_OFFLOAD_OBJECT_LIBS)
    set(CSAN_OFFLOAD_VERSION_LIBS)
    if(TARGET RTCsan_offload.${arch} AND TARGET RTSanitizerOffload.${arch})
      set(CSAN_OFFLOAD_OBJECT_LIBS RTCsan_offload RTSanitizerOffload)
      set(CSAN_OFFLOAD_VERSION_LIBS clang_rt.csan_offload-${arch})
    endif()

    add_sanitizer_rt_version_list(clang_rt.csan-dynamic-${arch}
                                  LIBS clang_rt.csan-${arch}
                                       ${CSAN_OFFLOAD_VERSION_LIBS}
                                  EXTRA csan.syms.extra)
    set(VERSION_SCRIPT_FLAG
        -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/clang_rt.csan-dynamic-${arch}.vers)
    set_property(SOURCE
      ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
      APPEND PROPERTY
      OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/clang_rt.csan-dynamic-${arch}.vers)

    add_compiler_rt_runtime(clang_rt.csan
      SHARED
      ARCHS ${arch}
      OBJECT_LIBS RTCsan
                  ${CSAN_COMMON_RUNTIME_OBJECT_LIBS}
                  ${CSAN_OFFLOAD_OBJECT_LIBS}
                  RTCsan_dynamic_version_script_dummy
      CFLAGS ${CSAN_CFLAGS}
      LINK_FLAGS ${CSAN_LINK_FLAGS} ${VERSION_SCRIPT_FLAG}
      LINK_LIBS ${CSAN_DYNAMIC_LIBS}
      PARENT_TARGET csan)
  endforeach()
endif()
