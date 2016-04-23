# Build for the EfficiencySanitizer runtime support library.

add_custom_target(esan)

set(ESAN_RTL_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF ESAN_RTL_CFLAGS)

include_directories(..)

set(ESAN_SOURCES
  esan.cpp
  esan_interface.cpp
  esan_interceptors.cpp)

foreach (arch ${ESAN_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.esan
    STATIC
    ARCHS ${arch}
    SOURCES ${ESAN_SOURCES}
            $<TARGET_OBJECTS:RTInterception.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
            $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
    CFLAGS ${ESAN_RTL_CFLAGS})
  add_sanitizer_rt_symbols(clang_rt.esan
    ARCHS ${arch}
    EXTRA esan.syms.extra)
  add_dependencies(esan
    clang_rt.esan-${arch}
    clang_rt.esan-${arch}-symbols)
endforeach()

add_dependencies(compiler-rt esan)

if (COMPILER_RT_INCLUDE_TESTS)
  # TODO(bruening): add tests via add_subdirectory(tests)
endif()
