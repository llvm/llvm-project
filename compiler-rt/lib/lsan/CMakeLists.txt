include_directories(..)

set(LSAN_CFLAGS
  ${SANITIZER_COMMON_CFLAGS}
  -fno-rtti)

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

# The common files need to build on every arch supported by ASan.
# (Even if they build into dummy object files.)
filter_available_targets(LSAN_COMMON_SUPPORTED_ARCH
  x86_64 i386 powerpc64)

# Architectures supported by the standalone LSan.
filter_available_targets(LSAN_SUPPORTED_ARCH
  x86_64)

set(LSAN_RUNTIME_LIBRARIES)

if(APPLE)
  foreach(os ${SANITIZER_COMMON_SUPPORTED_DARWIN_OS})
    add_compiler_rt_darwin_object_library(RTLSanCommon ${os}
      ARCH ${LSAN_COMMON_SUPPORTED_ARCH}
      SOURCES ${LSAN_COMMON_SOURCES}
      CFLAGS ${LSAN_CFLAGS})
  endforeach()
elseif(NOT ANDROID)
  foreach(arch ${LSAN_COMMON_SUPPORTED_ARCH})
    add_compiler_rt_object_library(RTLSanCommon ${arch}
      SOURCES ${LSAN_COMMON_SOURCES}
      CFLAGS ${LSAN_CFLAGS})
  endforeach()

  foreach(arch ${LSAN_SUPPORTED_ARCH})
    add_compiler_rt_static_runtime(clang_rt.lsan-${arch} ${arch}
      SOURCES ${LSAN_SOURCES}
              $<TARGET_OBJECTS:RTInterception.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommon.${arch}>
              $<TARGET_OBJECTS:RTSanitizerCommonLibc.${arch}>
              $<TARGET_OBJECTS:RTLSanCommon.${arch}>
      CFLAGS ${LSAN_CFLAGS})
    list(APPEND LSAN_RUNTIME_LIBRARIES clang_rt.lsan-${arch})
  endforeach()
endif()

if (LLVM_INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
add_subdirectory(lit_tests)
