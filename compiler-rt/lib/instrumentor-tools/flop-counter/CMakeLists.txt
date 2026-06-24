# CMakeLists.txt for FLOP Counter Example
#
# This example demonstrates counting floating-point operations using the
# Instrumentor pass. It provides a runtime library that can be linked with
# instrumented code to track and report FLOP counts.

add_compiler_rt_component(flop-counter)

set(FLOP_COUNTER_SOURCES
  flop_counter_runtime.cpp
  )

set(FLOP_COUNTER_HEADERS
  )

# Include paths for instrumentor runtime headers
# The instrumentor runtime headers are in llvm/utils
include_directories(${COMPILER_RT_SOURCE_DIR}/../llvm/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Common flags
set(FLOP_COUNTER_CFLAGS
  ${COMPILER_RT_COMMON_CFLAGS}
  -std=c++17
  )

# Determine supported architectures
if(APPLE)
  # On Darwin, use the darwin OSX architectures
  set(FLOP_COUNTER_SUPPORTED_ARCH arm64)
  if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
    set(FLOP_COUNTER_SUPPORTED_ARCH ${CMAKE_OSX_ARCHITECTURES})
  endif()
  if(DARWIN_osx_ARCHS)
    set(FLOP_COUNTER_SUPPORTED_ARCH ${DARWIN_osx_ARCHS})
  endif()
else()
  # For non-Apple platforms, use the default target architecture
  set(FLOP_COUNTER_SUPPORTED_ARCH ${COMPILER_RT_DEFAULT_TARGET_ARCH})
endif()

message(STATUS "FLOP Counter supported architectures: ${FLOP_COUNTER_SUPPORTED_ARCH}")

# Build the static runtime library for Apple platforms
if(APPLE)
  add_compiler_rt_runtime(clang_rt.instrumentor_flop_counter
    STATIC
    OS osx
    ARCHS ${FLOP_COUNTER_SUPPORTED_ARCH}
    CFLAGS ${FLOP_COUNTER_CFLAGS}
    SOURCES ${FLOP_COUNTER_SOURCES}
    ADDITIONAL_HEADERS ${FLOP_COUNTER_HEADERS}
    PARENT_TARGET flop-counter)
else()
  add_compiler_rt_runtime(clang_rt.instrumentor_flop_counter
    STATIC
    ARCHS ${FLOP_COUNTER_SUPPORTED_ARCH}
    CFLAGS ${FLOP_COUNTER_CFLAGS}
    SOURCES ${FLOP_COUNTER_SOURCES}
    ADDITIONAL_HEADERS ${FLOP_COUNTER_HEADERS}
    PARENT_TARGET flop-counter)
endif()

# Install the configuration file as a resource
install(FILES flop_counter_config.json
        DESTINATION share/llvm/instrumentor-configs
        COMPONENT flop-counter)
