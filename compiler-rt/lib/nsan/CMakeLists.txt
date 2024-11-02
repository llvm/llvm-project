add_compiler_rt_component(nsan)

include_directories(..)

set(NSAN_SOURCES
  nsan.cpp
  nsan_flags.cpp
  nsan_interceptors.cpp
  nsan_stats.cpp
  nsan_suppressions.cpp
)

set(NSAN_HEADERS
  nsan.h
  nsan_flags.h
  nsan_flags.inc
  nsan_platform.h
  nsan_stats.h
  nsan_suppressions.h
)

append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC NSAN_CFLAGS)

set(NSAN_DYNAMIC_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

set(NSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})

if (COMPILER_RT_HAS_NSAN)
  foreach(arch ${NSAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(
      clang_rt.nsan
      STATIC
      ARCHS ${arch}
      SOURCES ${NSAN_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonCoverage.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonSymbolizer.${arch}>
              $<TARGET_OBJECTS:RTUbsan.${arch}>
      ADDITIONAL_HEADERS ${NSAN_HEADERS}
      CFLAGS ${NSAN_CFLAGS}
      PARENT_TARGET nsan
    )
  endforeach()

  add_compiler_rt_object_libraries(RTNsan
      ARCHS ${NSAN_SUPPORTED_ARCH}
      SOURCES ${NSAN_SOURCES}
      ADDITIONAL_HEADERS ${NSAN_HEADERS}
      CFLAGS ${NSAN_CFLAGS})
endif()

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
