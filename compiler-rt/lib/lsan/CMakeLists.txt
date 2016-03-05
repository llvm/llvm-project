include_directories(..)

set(LSAN_CFLAGS ${SANITIZER_COMMON_CFLAGS})
append_rtti_flag(OFF LSAN_CFLAGS)

set(LSAN_COMMON_SOURCES
  lsan_common.cc
  lsan_common_linux.cc)

set(LSAN_SOURCES
  lsan.cc
  lsan_allocator.cc
  lsan_interceptors.cc
  lsan_preinit.cc
  lsan_thread.cc)

set(LSAN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(lsan)

add_compiler_rt_object_libraries(RTLSanCommon
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${LSAN_COMMON_SUPPORTED_ARCH}
    SOURCES ${LSAN_COMMON_SOURCES}
    CFLAGS ${LSAN_CFLAGS})

if(COMPILER_RT_HAS_LSAN)
  foreach(arch ${LSAN_SUPPORTED_ARCH})
    add_compiler_rt_runtime(clang_rt.lsan
      STATIC
      ARCHS ${arch}
      SOURCES ${LSAN_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTLSanCommon.${arch}>
      CFLAGS ${LSAN_CFLAGS}
      PARENT_TARGET lsan)
  endforeach()
endif()

add_dependencies(compiler-rt lsan)
