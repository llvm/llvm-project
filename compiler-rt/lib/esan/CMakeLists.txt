# Build for the EfficiencySanitizer runtime support library.

add_compiler_rt_component(esan)

set(ESAN_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF ESAN_RTL_CFLAGS)

include_directories(..)

set(ESAN_SOURCES
  esan.cpp
  esan_flags.cpp
  esan_interface.cpp
  esan_interceptors.cpp
  esan_linux.cpp
  esan_sideline_linux.cpp
  esan_sideline_bsd.cpp
  cache_frag.cpp
  working_set.cpp
  working_set_posix.cpp)

set(ESAN_HEADERS
  cache_frag.h
  esan.h
  esan_circular_buffer.h
  esan_flags.h
  esan_flags.inc
  esan_hashtable.h
  esan_interface_internal.h
  esan_shadow.h
  esan_sideline.h
  working_set.h)

foreach (arch ${ESAN_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.esan
    STATIC
    ARCHS ${arch}
    SOURCES ${ESAN_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
    ADDITIONAL_HEADERS ${ESAN_HEADERS}
    CFLAGS ${ESAN_RTL_CFLAGS})
  add_sanitizer_rt_symbols(clang_rt.esan
    ARCHS ${arch}
    EXTRA esan.syms.extra)
  add_dependencies(esan
    clang_rt.esan-${arch}
    clang_rt.esan-${arch}-symbols)
endforeach()

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
