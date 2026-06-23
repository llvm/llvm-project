# CMakeLists.txt for Pointer Tracking Example
#
# This example demonstrates tracking pointer allocations and usage patterns using
# the Instrumentor pass. It provides a runtime library that replaces pointers with
# fake pointers encoding allocation IDs and tracks read/write patterns and timing.

add_compiler_rt_component(pointer-tracking)

set(POINTER_TRACKING_SOURCES
  pointer_tracking_runtime.cpp
  )

set(POINTER_TRACKING_HEADERS
  )

# Include paths for instrumentor runtime headers
# The instrumentor runtime headers are in llvm/utils
include_directories(${COMPILER_RT_SOURCE_DIR}/../llvm/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Common flags
set(POINTER_TRACKING_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++17
  )

# Determine supported architectures
if(APPLE)
  # On Darwin, use the darwin OSX architectures
  set(POINTER_TRACKING_SUPPORTED_ARCH arm64)
  if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
    set(POINTER_TRACKING_SUPPORTED_ARCH ${CMAKE_OSX_ARCHITECTURES})
  endif()
  if(DARWIN_osx_ARCHS)
    set(POINTER_TRACKING_SUPPORTED_ARCH ${DARWIN_osx_ARCHS})
  endif()
else()
  # For non-Apple platforms, use the default target architecture
  set(POINTER_TRACKING_SUPPORTED_ARCH ${COMPILER_RT_DEFAULT_TARGET_ARCH})
endif()

message(STATUS "Pointer Tracking supported architectures: ${POINTER_TRACKING_SUPPORTED_ARCH}")

# Build the static runtime library for Apple platforms
if(APPLE)
  add_compiler_rt_runtime(clang_rt.instrumentor_pointer_tracking
    STATIC
    OS osx
    ARCHS ${POINTER_TRACKING_SUPPORTED_ARCH}
    CFLAGS ${POINTER_TRACKING_CFLAGS}
    SOURCES ${POINTER_TRACKING_SOURCES}
    ADDITIONAL_HEADERS ${POINTER_TRACKING_HEADERS}
    PARENT_TARGET pointer-tracking)
else()
  add_compiler_rt_runtime(clang_rt.instrumentor_pointer_tracking
    STATIC
    ARCHS ${POINTER_TRACKING_SUPPORTED_ARCH}
    CFLAGS ${POINTER_TRACKING_CFLAGS}
    SOURCES ${POINTER_TRACKING_SOURCES}
    ADDITIONAL_HEADERS ${POINTER_TRACKING_HEADERS}
    PARENT_TARGET pointer-tracking)
endif()

# Install the configuration file as a resource
install(FILES pointer_tracking_config.json
        DESTINATION share/llvm/instrumentor-configs
        COMPONENT pointer-tracking)
