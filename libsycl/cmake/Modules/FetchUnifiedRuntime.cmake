#===============================================================================
# Fetches Unified Runtime used by SYCL language runtime to abstract SYCL open 
# standard and vendor heterogeneous offload interfaces.
#
# This will in time be replaced by the new LLVM Offload interface.
#
# Unified Runtime is Apache 2.0 license with LLVM exceptions. 
#
#===============================================================================

option(LIBSYCL_UR_BUILD_TESTS "Build tests for UR" OFF)

set(UR_BUILD_TESTS "${LIBSYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)
# UR tests require the examples to be built
set(UR_BUILD_EXAMPLES "${LIBSYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)

if("level_zero" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_L0 ON)
endif()
if("cuda" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_CUDA ON)
endif()
if("hip" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_HIP ON)
endif()
if("opencl" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  set(UR_BUILD_ADAPTER_OPENCL ON)
endif()

# Disable errors from warnings while building the UR.
# And remember the original flags before doing that.
set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
if(WIN32)
  append("/WX-" CMAKE_CXX_FLAGS)
  append("/WX-" CMAKE_C_FLAGS)
  # Unified runtime build fails with /DUNICODE
  append("/UUNICODE" CMAKE_CXX_FLAGS)
  append("/UUNICODE" CMAKE_C_FLAGS)
  append("/EHsc" CMAKE_CXX_FLAGS)
  append("/EHsc" CMAKE_C_FLAGS)
else()
  append("-Wno-error" CMAKE_CXX_FLAGS)
  append("-Wno-error" CMAKE_C_FLAGS)
endif()

if(NOT FETCHCONTENT_SOURCE_DIR_UNIFIED-RUNTIME)
  find_package(unified-runtime)
  if(unified-runtime_FOUND)
    message (STATUS "Found system install of unified-runtime")
    return()
  endif()
endif()

include(FetchContent)

set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
set(UNIFIED_RUNTIME_TAG 8ee4da175c197d4dc5c9ec939e7e4d87d4edfa99)

FetchContent_Declare(unified-runtime
  GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
  GIT_TAG           ${UNIFIED_RUNTIME_TAG}
)

FetchContent_GetProperties(unified-runtime)
if(FETCHCONTENT_SOURCE_DIR_UNIFIED-RUNTIME)
  message(STATUS "Using specified Unified Runtime repo location at ${FETCHCONTENT_SOURCE_DIR_UNIFIED-RUNTIME}")
else()
  message(STATUS "Cloning Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
endif()
FetchContent_MakeAvailable(unified-runtime)

set(UNIFIED_RUNTIME_SOURCE_DIR
  "${unified-runtime_SOURCE_DIR}" CACHE PATH
  "Path to Unified Runtime Headers" FORCE)

set(UMF_BUILD_EXAMPLES OFF CACHE INTERNAL "UMF EXAMPLES")
# Due to the use of dependentloadflag and no installer for UMF and hwloc we need
# to link statically on windows
if(WIN32)
  set(UMF_BUILD_SHARED_LIBRARY OFF CACHE INTERNAL "Build UMF shared library")
  set(UMF_LINK_HWLOC_STATICALLY ON CACHE INTERNAL "static HWLOC")
endif()

# Restore original flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

message(STATUS
  "Using Unified Runtime source directory: ${UNIFIED_RUNTIME_SOURCE_DIR}")

set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
set(UNIFIED_RUNTIME_SRC_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source")
set(UNIFIED_RUNTIME_COMMON_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source/common")

add_library(UnifiedRuntimeLoader ALIAS ur_loader)
add_library(UnifiedRuntime-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

add_custom_target(UnifiedRuntimeAdapters)

function(add_sycl_ur_adapter NAME)
  add_dependencies(UnifiedRuntimeAdapters ur_adapter_${NAME})
endfunction()

if("level_zero" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(level_zero)
endif()

if("cuda" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(cuda)
endif()

if("hip" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(hip)
endif()

if("opencl" IN_LIST LIBSYCL_ENABLE_BACKENDS)
  add_sycl_ur_adapter(opencl)
endif()
