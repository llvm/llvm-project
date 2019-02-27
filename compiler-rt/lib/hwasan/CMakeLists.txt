include_directories(..)

# Runtime library sources and build flags.
set(HWASAN_RTL_SOURCES
  hwasan.cpp
  hwasan_allocator.cpp
  hwasan_dynamic_shadow.cpp
  hwasan_interceptors.cpp
  hwasan_interceptors_vfork.S
  hwasan_linux.cpp
  hwasan_memintrinsics.cpp
  hwasan_poisoning.cpp
  hwasan_report.cpp
  hwasan_thread.cpp
  hwasan_thread_list.cpp
  )

set(HWASAN_RTL_CXX_SOURCES
  hwasan_new_delete.cpp
  )

set(HWASAN_RTL_HEADERS
  hwasan.h
  hwasan_allocator.h
  hwasan_dynamic_shadow.h
  hwasan_flags.h
  hwasan_flags.inc
  hwasan_interface_internal.h
  hwasan_malloc_bisect.h
  hwasan_mapping.h
  hwasan_poisoning.h
  hwasan_report.h
  hwasan_thread.h
  hwasan_thread_list.h
  )

set(HWASAN_DEFINITIONS)
append_list_if(COMPILER_RT_HWASAN_WITH_INTERCEPTORS HWASAN_WITH_INTERCEPTORS=1 HWASAN_DEFINITIONS)

set(HWASAN_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF HWASAN_RTL_CFLAGS)
append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC HWASAN_RTL_CFLAGS)
# Prevent clang from generating libc calls.
append_list_if(COMPILER_RT_HAS_FFREESTANDING_FLAG -ffreestanding HWASAN_RTL_CFLAGS)

set(HWASAN_DYNAMIC_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

if(ANDROID)
# Put most Sanitizer shared libraries in the global group. For more details, see
# android-changes-for-ndk-developers.md#changes-to-library-search-order
  if (COMPILER_RT_HAS_Z_GLOBAL)
    list(APPEND HWASAN_DYNAMIC_LINK_FLAGS -Wl,-z,global)
  endif()
endif()

set(HWASAN_DYNAMIC_CFLAGS ${HWASAN_RTL_CFLAGS})
append_list_if(COMPILER_RT_HAS_FTLS_MODEL_INITIAL_EXEC
  -ftls-model=initial-exec HWASAN_DYNAMIC_CFLAGS)
append_list_if(MSVC /DEBUG HWASAN_DYNAMIC_LINK_FLAGS)

set(HWASAN_DYNAMIC_LIBS ${SANITIZER_CXX_ABI_LIBRARIES} ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl HWASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt HWASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m HWASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread HWASAN_DYNAMIC_LIBS)

if (TARGET cxx-headers OR HAVE_LIBCXX)
  set(HWASAN_DEPS cxx-headers)
endif()

# Static runtime library.
add_compiler_rt_component(hwasan)

add_compiler_rt_object_libraries(RTHwasan
  ARCHS ${HWASAN_SUPPORTED_ARCH}
  SOURCES ${HWASAN_RTL_SOURCES}
  ADDITIONAL_HEADERS ${HWASAN_RTL_HEADERS}
  CFLAGS ${HWASAN_RTL_CFLAGS}
  DEFS ${HWASAN_DEFINITIONS}
  DEPS ${HWASAN_DEPS})
add_compiler_rt_object_libraries(RTHwasan_cxx
  ARCHS ${HWASAN_SUPPORTED_ARCH}
  SOURCES ${HWASAN_RTL_CXX_SOURCES}
  ADDITIONAL_HEADERS ${HWASAN_RTL_HEADERS}
  CFLAGS ${HWASAN_RTL_CFLAGS}
  DEFS ${HWASAN_DEFINITIONS}
  DEPS ${HWASAN_DEPS})
add_compiler_rt_object_libraries(RTHwasan_dynamic
  ARCHS ${HWASAN_SUPPORTED_ARCH}
  SOURCES ${HWASAN_RTL_SOURCES} ${HWASAN_RTL_CXX_SOURCES}
  ADDITIONAL_HEADERS ${HWASAN_RTL_HEADERS}
  CFLAGS ${HWASAN_DYNAMIC_CFLAGS}
  DEFS ${HWASAN_DEFINITIONS}
  DEPS ${HWASAN_DEPS})

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc "")
add_compiler_rt_object_libraries(RTHwasan_dynamic_version_script_dummy
  ARCHS ${HWASAN_SUPPORTED_ARCH}
  SOURCES ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc
  CFLAGS ${HWASAN_DYNAMIC_CFLAGS}
  DEFS ${HWASAN_DEFINITIONS}
  DEPS ${HWASAN_DEPS})

foreach(arch ${HWASAN_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.hwasan
    STATIC
    ARCHS ${arch}
    OBJECT_LIBS RTHwasan
                RTInterception
                RTSanitizerCommon
                RTSanitizerCommonLibc
                RTSanitizerCommonCoverage
                RTSanitizerCommonSymbolizer
                RTUbsan
    CFLAGS ${HWASAN_RTL_CFLAGS}
    PARENT_TARGET hwasan)
  add_compiler_rt_runtime(clang_rt.hwasan_cxx
    STATIC
    ARCHS ${arch}
    OBJECT_LIBS RTHwasan_cxx
                RTUbsan_cxx
    CFLAGS ${HWASAN_RTL_CFLAGS}
    PARENT_TARGET hwasan)

  if (UNIX)
    add_sanitizer_rt_version_list(clang_rt.hwasan-dynamic-${arch}
                                  LIBS clang_rt.hwasan-${arch} clang_rt.hwasan_cxx-${arch}
                                  EXTRA hwasan.syms.extra)
    set(VERSION_SCRIPT_FLAG
         -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/clang_rt.hwasan-dynamic-${arch}.vers)
    set_property(SOURCE
      ${CMAKE_CURRENT_BINARY_DIR}/dummy.cc
      APPEND PROPERTY
      OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/clang_rt.hwasan-dynamic-${arch}.vers)
  else()
    set(VERSION_SCRIPT_FLAG)
  endif()


  add_compiler_rt_runtime(clang_rt.hwasan
    SHARED
    ARCHS ${arch}
    OBJECT_LIBS
            RTHwasan_dynamic
            RTInterception
            RTSanitizerCommon
            RTSanitizerCommonLibc
            RTSanitizerCommonCoverage
            RTSanitizerCommonSymbolizer
            RTUbsan
            RTUbsan_cxx
            # The only purpose of RTHWAsan_dynamic_version_script_dummy is to
            # carry a dependency of the shared runtime on the version script.
            # Replacing it with a straightforward
            # add_dependencies(clang_rt.asan-dynamic-${arch} clang_rt.asan-dynamic-${arch}-version-list)
            # generates an order-only dependency in ninja.
            RTHwasan_dynamic_version_script_dummy
    CFLAGS ${HWASAN_DYNAMIC_CFLAGS}
    LINK_FLAGS ${HWASAN_DYNAMIC_LINK_FLAGS}
              ${VERSION_SCRIPT_FLAG}
    LINK_LIBS ${HWASAN_DYNAMIC_LIBS}
    DEFS ${ASAN_DYNAMIC_DEFINITIONS}
    PARENT_TARGET hwasan)

  if(SANITIZER_USE_SYMBOLS)
    add_sanitizer_rt_symbols(clang_rt.hwasan
      ARCHS ${arch}
      EXTRA hwasan.syms.extra)
    add_sanitizer_rt_symbols(clang_rt.hwasan_cxx
      ARCHS ${arch}
      EXTRA hwasan.syms.extra)
    add_dependencies(hwasan clang_rt.hwasan-${arch}-symbols
                          clang_rt.hwasan_cxx-${arch}-symbols)
  endif()
endforeach()

add_compiler_rt_resource_file(hwasan_blacklist hwasan_blacklist.txt hwasan)

# if(COMPILER_RT_INCLUDE_TESTS)
#   add_subdirectory(tests)
# endif()
