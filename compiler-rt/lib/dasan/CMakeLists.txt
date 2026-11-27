include_directories(..)

include(FindLibcCommonUtils)

# The GPU half only reports failed accesses over RPC.
if(COMPILER_RT_GPU_BUILD)
  if(NOT TARGET llvm-libc-common-utilities)
    add_compiler_rt_component(dasan)
    return()
  endif()

  set(DASAN_DEVICE_HEADERS
    dasan_offload_packet.h
    )

  set(DASAN_DEVICE_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS)
  append_rtti_flag(OFF DASAN_DEVICE_CFLAGS)

  add_compiler_rt_component(dasan)

  add_compiler_rt_runtime(clang_rt.dasan
    STATIC
    ARCHS ${DASAN_SUPPORTED_ARCH}
    SOURCES dasan_device.cpp
    ADDITIONAL_HEADERS ${DASAN_DEVICE_HEADERS}
    CFLAGS ${DASAN_DEVICE_CFLAGS}
    LINK_LIBS llvm-libc-common-utilities
    PARENT_TARGET dasan)

  return()
endif()

if(NOT OS_NAME MATCHES "Linux" OR NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(DASAN_SOURCES
  dasan.cpp
  dasan_allocator.cpp
  dasan_host.cpp
  dasan_hsa_interceptors.cpp
  dasan_image.cpp
  dasan_platform.cpp
  dasan_report.cpp
  )

set(DASAN_HEADERS
  dasan.h
  dasan_allocator.h
  dasan_flags.h
  dasan_flags.inc
  dasan_image.h
  dasan_mapping.h
  dasan_offload_packet.h
  dasan_platform.h
  dasan_report.h
  )

set(DASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF DASAN_CFLAGS)

set(DASAN_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

set(DASAN_DYNAMIC_LIBS
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_CXX_ABI_LIBRARIES}
  ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl DASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt DASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread DASAN_DYNAMIC_LIBS)

add_compiler_rt_object_libraries(RTDasan
  ARCHS ${DASAN_SUPPORTED_ARCH}
  SOURCES ${DASAN_SOURCES}
  ADDITIONAL_HEADERS ${DASAN_HEADERS}
  CFLAGS ${DASAN_CFLAGS})
foreach(arch ${DASAN_SUPPORTED_ARCH})
  target_link_libraries(RTDasan.${arch} PRIVATE llvm-libc-common-utilities)
endforeach()

add_compiler_rt_component(dasan)

add_compiler_rt_runtime(clang_rt.dasan
  STATIC
  ARCHS ${DASAN_SUPPORTED_ARCH}
  OBJECT_LIBS RTDasan
              RTSanitizerOffload
              RTInterception
              RTSanitizerCommon
              RTSanitizerCommonLibc
              RTSanitizerCommonCoverage
              RTSanitizerCommonSymbolizer
  CFLAGS ${DASAN_CFLAGS}
  PARENT_TARGET dasan)

add_compiler_rt_runtime(clang_rt.dasan-preinit
  STATIC
  ARCHS ${DASAN_SUPPORTED_ARCH}
  SOURCES dasan_preinit.cpp
  ADDITIONAL_HEADERS ${DASAN_HEADERS}
  CFLAGS ${DASAN_CFLAGS}
  PARENT_TARGET dasan)

if(COMPILER_RT_HAS_VERSION_SCRIPT)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
  add_compiler_rt_object_libraries(RTDasan_dynamic_version_script_dummy
    ARCHS ${DASAN_SUPPORTED_ARCH}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
    CFLAGS ${DASAN_CFLAGS})

  foreach(arch ${DASAN_SUPPORTED_ARCH})
    add_sanitizer_rt_version_list(clang_rt.dasan-dynamic-${arch}
                                  LIBS clang_rt.dasan-${arch}
                                  EXTRA dasan.syms.extra)
    set(VERSION_SCRIPT_FLAG
        -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/clang_rt.dasan-dynamic-${arch}.vers)
    if(COMPILER_RT_HAS_GNU_VERSION_SCRIPT_COMPAT)
      list(APPEND VERSION_SCRIPT_FLAG -Wl,-z,gnu-version-script-compat)
    endif()
    set_property(SOURCE
      ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp
      APPEND PROPERTY
      OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/clang_rt.dasan-dynamic-${arch}.vers)

    add_compiler_rt_runtime(clang_rt.dasan
      SHARED
      ARCHS ${arch}
      OBJECT_LIBS RTDasan
                  RTSanitizerOffload
                  RTInterception
                  RTSanitizerCommon
                  RTSanitizerCommonLibc
                  RTSanitizerCommonCoverage
                  RTSanitizerCommonSymbolizer
                  RTDasan_dynamic_version_script_dummy
      CFLAGS ${DASAN_CFLAGS}
      LINK_FLAGS ${DASAN_LINK_FLAGS} ${VERSION_SCRIPT_FLAG}
      LINK_LIBS ${DASAN_DYNAMIC_LIBS}
      PARENT_TARGET dasan)
  endforeach()
endif()

# The list of symbols the driver hands the linker as --dynamic-list.
if(SANITIZER_USE_SYMBOLS)
  add_sanitizer_rt_symbols(clang_rt.dasan
    ARCHS ${DASAN_SUPPORTED_ARCH}
    PARENT_TARGET dasan
    EXTRA dasan.syms.extra)
endif()
