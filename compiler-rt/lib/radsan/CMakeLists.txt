include_directories(..)

set(RADSAN_CXX_SOURCES
  radsan.cpp
  radsan_context.cpp
  radsan_stack.cpp
  radsan_interceptors.cpp)

set(RADSAN_PREINIT_SOURCES
  radsan_preinit.cpp)

set(RADSAN_HEADERS
  radsan.h
  radsan_context.h
  radsan_stack.h)

set(RADSAN_DEPS)

set(RADSAN_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  ${COMPILER_RT_CXX_CFLAGS}
  -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS)
set(RADSAN_LINK_FLAGS ${COMPILER_RT_COMMON_LINK_FLAGS})
set(RADSAN_LINK_LIBS
  ${COMPILER_RT_UNWINDER_LINK_LIBS}
  ${COMPILER_RT_CXX_LINK_LIBS})

if(APPLE)
  add_compiler_rt_object_libraries(RTRadsan
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${RADSAN_SUPPORTED_ARCH}
    SOURCES ${RADSAN_CXX_SOURCES}
    ADDITIONAL_HEADERS ${RADSAN_HEADERS}
    CFLAGS ${RADSAN_CFLAGS}
    DEPS ${RADSAN_DEPS})
else()
  add_compiler_rt_object_libraries(RTRadsan
    ARCHS ${RADSAN_SUPPORTED_ARCH}
    SOURCES ${RADSAN_CXX_SOURCES}
    ADDITIONAL_HEADERS ${RADSAN_HEADERS}
    CFLAGS ${RADSAN_CFLAGS}
    DEPS ${RADSAN_DEPS})
  add_compiler_rt_object_libraries(RTRadsan_preinit
    ARCHS ${RADSAN_SUPPORTED_ARCH}
    SOURCES ${RADSAN_PREINIT_SOURCES}
    ADDITIONAL_HEADERS ${RADSAN_HEADERS}
    CFLAGS ${RADSAN_CFLAGS})
endif()

set(RADSAN_COMMON_RUNTIME_OBJECT_LIBS
  RTInterception
  RTSanitizerCommon
  RTSanitizerCommonLibc
  RTSanitizerCommonCoverage
  RTSanitizerCommonSymbolizer)

append_list_if(COMPILER_RT_HAS_LIBDL dl RADSAN_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBRT rt RADSAN_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBM m RADSAN_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBPTHREAD pthread RADSAN_LINK_LIBS)
append_list_if(COMPILER_RT_HAS_LIBLOG log RADSAN_LINK_LIBS)

add_compiler_rt_component(radsan)

if (APPLE)
  add_weak_symbols("sanitizer_common" WEAK_SYMBOL_LINK_FLAGS)
  set(RADSAN_LINK_FLAGS ${RADSAN_LINK_FLAGS} ${WEAK_SYMBOL_LINK_FLAGS})

  add_compiler_rt_runtime(clang_rt.radsan
    SHARED
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${RADSAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTRadsan
                ${RADSAN_COMMON_RUNTIME_OBJECT_LIBS}
    LINK_FLAGS ${RADSAN_LINK_FLAGS}
    LINK_LIBS ${RADSAN_LINK_LIBS}
    PARENT_TARGET radsan)
else()
  add_compiler_rt_runtime(clang_rt.radsan
    STATIC
    ARCHS ${RADSAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTRadsan_preinit
                RTRadsan
                ${RADSAN_COMMON_RUNTIME_OBJECT_LIBS}
    LINK_FLAGS ${RADSAN_LINK_FLAGS}
    CFLAGS ${RADSAN_CFLAGS}
    PARENT_TARGET radsan)
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
