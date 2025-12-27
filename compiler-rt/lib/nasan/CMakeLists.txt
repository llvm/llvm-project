# Build for the NoAliasSanitizer runtime library.

add_compiler_rt_component(nasan)

include_directories(..)

set(NASAN_SOURCES
  nasan_rtl.cpp
  nasan_shadow.cpp
  nasan_report.cpp
  nasan_interceptors.cpp
  nasan_stack.cpp
)

set(NASAN_HEADERS
  nasan.h
  nasan_internal.h
)

set(NASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF NASAN_CFLAGS)

# Remove -fno-builtin to allow our interceptors to work
list(FILTER NASAN_CFLAGS EXCLUDE REGEX "-fno-builtin")

set(NASAN_DYNAMIC_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

set(NASAN_COMMON_RUNTIME_OBJECT_LIBS
  RTInterception
  RTSanitizerCommon
  RTSanitizerCommonLibc
  RTSanitizerCommonCoverage
  RTSanitizerCommonSymbolizer)

set(NASAN_DYNAMIC_LIBS
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${SANITIZER_CXX_ABI_LIBRARIES}
  ${SANITIZER_COMMON_LINK_LIBS})

append_list_if(COMPILER_RT_HAS_LIBDL dl NASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt NASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m NASAN_DYNAMIC_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread NASAN_DYNAMIC_LIBS)

# Compile sources into object libraries
add_compiler_rt_object_libraries(RTNasan_dynamic
  OS ${SANITIZER_COMMON_SUPPORTED_OS}
  ARCHS ${NASAN_SUPPORTED_ARCH}
  SOURCES ${NASAN_SOURCES}
  ADDITIONAL_HEADERS ${NASAN_HEADERS}
  CFLAGS ${NASAN_CFLAGS})

if(APPLE)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)

  add_compiler_rt_runtime(clang_rt.nasan
    SHARED
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${NASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTNasan_dynamic
                ${NASAN_COMMON_RUNTIME_OBJECT_LIBS}
    CFLAGS ${NASAN_CFLAGS}
    LINK_FLAGS ${WEAK_SYMBOL_LINK_FLAGS} ${NASAN_DYNAMIC_LINK_FLAGS}
    LINK_LIBS ${NASAN_DYNAMIC_LIBS}
    PARENT_TARGET nasan)
else()
  add_compiler_rt_object_libraries(RTNasan
    ARCHS ${NASAN_SUPPORTED_ARCH}
    SOURCES ${NASAN_SOURCES}
    ADDITIONAL_HEADERS ${NASAN_HEADERS}
    CFLAGS ${NASAN_CFLAGS})

  # Build static runtime
  add_compiler_rt_runtime(clang_rt.nasan
    STATIC
    ARCHS ${NASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTNasan
                ${NASAN_COMMON_RUNTIME_OBJECT_LIBS}
    CFLAGS ${NASAN_CFLAGS}
    PARENT_TARGET nasan)

  # Build shared runtime for each architecture
  foreach(arch ${NASAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.nasan
      SHARED
      ARCHS ${arch}
      OBJECT_LIBS RTNasan_dynamic
                  ${NASAN_COMMON_RUNTIME_OBJECT_LIBS}
      CFLAGS ${NASAN_CFLAGS}
      LINK_FLAGS ${NASAN_DYNAMIC_LINK_FLAGS}
      LINK_LIBS ${NASAN_DYNAMIC_LIBS}
      PARENT_TARGET nasan)
  endforeach()
endif()

# TODO: Add tests
# if(COMPILER_RT_INCLUDE_TESTS)
#   add_subdirectory(tests)
# endif()
