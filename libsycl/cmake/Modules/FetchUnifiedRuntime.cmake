#===============================================================================
# Fetches Unified Runtime used by SYCL language runtime to abstract SYCL open 
# standard and vendor heterogeneous offload interfaces.
#
# This will in time be replaced by the new LLVM Offload interface.
#
# Unified Runtime is Apache 2.0 license with LLVM exceptions. 
#
#===============================================================================

# Options to override the default behaviour of the FetchContent to include UR
# source code.
set(LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_REPO
  "" CACHE STRING "Override the Unified Runtime FetchContent repository")
set(LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_TAG
  "" CACHE STRING "Override the Unified Runtime FetchContent tag")

# Options to disable use of FetchContent to include Unified Runtime source code
# to improve developer workflow.
option(LIBSYCL_UR_USE_FETCH_CONTENT
  "Use FetchContent to acquire the Unified Runtime source code" ON)
set(LIBSYCL_UR_SOURCE_DIR
  "" CACHE PATH "Path to root of Unified Runtime repository")

option(LIBSYCL_UR_BUILD_TESTS "Build tests for UR" OFF)
set(UR_BUILD_TESTS "${LIBSYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)
# UR tests require the examples to be built
set(UR_BUILD_EXAMPLES "${LIBSYCL_UR_BUILD_TESTS}" CACHE BOOL "" FORCE)

set(UR_EXTERNAL_DEPENDENCIES "sycl-headers" CACHE STRING
  "List of external CMake targets for executables/libraries to depend on" FORCE)

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
# And remember origin flags before doing that.
set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
if(WIN32)
  append("/WX-" CMAKE_CXX_FLAGS)
  append("/WX-" CMAKE_C_FLAGS)
  # Unified runtime build fails with /DUNICODE
  append("/UUNICODE" CMAKE_CXX_FLAGS)
  append("/UUNICODE" CMAKE_C_FLAGS)
else()
  append("-Wno-error" CMAKE_CXX_FLAGS)
  append("-Wno-error" CMAKE_C_FLAGS)
endif()

if(LIBSYCL_UR_USE_FETCH_CONTENT)
  include(FetchContent)

  # The fetch_adapter_source function can be used to perform a separate content
  # fetch for a UR adapter (backend), this allows development of adapters to be decoupled
  # from each other.
  #
  # A separate content fetch will not be performed if:
  # * The adapter name is not present in the LIBSYCL_ENABLE_BACKENDS variable.
  # * The repo and tag provided match the values of the
  #   UNIFIED_RUNTIME_REPO/UNIFIED_RUNTIME_TAG variables
  #
  # Args:
  #   * name - Must be the directory name of the adapter
  #   * repo - A valid Git URL of a Unified Runtime repo
  #   * tag - A valid Git branch/tag/commit in the Unified Runtime repo
  function(fetch_adapter_source name repo tag)
    if(NOT ${name} IN_LIST LIBSYCL_ENABLE_BACKENDS)
      return()
    endif()
    if(repo STREQUAL UNIFIED_RUNTIME_REPO AND
        tag STREQUAL UNIFIED_RUNTIME_TAG)
      # If the adapter sources are taken from the main checkout, reset the
      # adapter specific source path.
      string(TOUPPER ${name} NAME)
      set(UR_ADAPTER_${NAME}_SOURCE_DIR ""
        CACHE PATH "Path to external '${name}' adapter source dir" FORCE)
      return()
    endif()
    message(STATUS
      "Will fetch Unified Runtime ${name} adapter from ${repo} at ${tag}")
    set(fetch-name ur-${name})
    FetchContent_Declare(${fetch-name}
      GIT_REPOSITORY ${repo} GIT_TAG ${tag})
    # We don't want to add this repo to the build, only fetch its source.
    FetchContent_Populate(${fetch-name})
    # Get the path to the source directory
    string(TOUPPER ${name} NAME)
    set(source_dir_var UR_ADAPTER_${NAME}_SOURCE_DIR)
    FetchContent_GetProperties(${fetch-name} SOURCE_DIR UR_ADAPTER_${NAME}_SOURCE_DIR)
    # Set the variable which informs UR where to get the adapter source from.
    set(UR_ADAPTER_${NAME}_SOURCE_DIR
      "${UR_ADAPTER_${NAME}_SOURCE_DIR}/source/adapters/${name}"
      CACHE PATH "Path to external '${name}' adapter source dir" FORCE)
  endfunction()

  set(UNIFIED_RUNTIME_REPO "https://github.com/oneapi-src/unified-runtime.git")
  set(UNIFIED_RUNTIME_TAG 851ee6a2bb5f5c34c6e48a00e0b06255044e0f03)

  set(UMF_BUILD_EXAMPLES OFF CACHE INTERNAL "EXAMPLES")
  # Due to the use of dependentloadflag and no installer for UMF and hwloc we need
  # to link statically on windows
  if(WIN32)
    set(UMF_BUILD_SHARED_LIBRARY OFF CACHE INTERNAL "Build UMF shared library")
    set(UMF_LINK_HWLOC_STATICALLY ON CACHE INTERNAL "static HWLOC")
  endif()

  fetch_adapter_source(level_zero
    ${UNIFIED_RUNTIME_REPO}
    ${UNIFIED_RUNTIME_TAG}
  )

  fetch_adapter_source(opencl
    ${UNIFIED_RUNTIME_REPO}
    ${UNIFIED_RUNTIME_TAG}
  )

  fetch_adapter_source(cuda
    ${UNIFIED_RUNTIME_REPO}
    ${UNIFIED_RUNTIME_TAG}
  )

  fetch_adapter_source(hip
    ${UNIFIED_RUNTIME_REPO}
    ${UNIFIED_RUNTIME_TAG}
  )

  if(LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_REPO)
    set(UNIFIED_RUNTIME_REPO "${LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_REPO}")
  endif()
  if(LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_TAG)
    set(UNIFIED_RUNTIME_TAG "${LIBSYCL_UR_OVERRIDE_FETCH_CONTENT_TAG}")
  endif()

  message(STATUS "Will fetch Unified Runtime from ${UNIFIED_RUNTIME_REPO}")
  FetchContent_Declare(unified-runtime
    GIT_REPOSITORY    ${UNIFIED_RUNTIME_REPO}
    GIT_TAG           ${UNIFIED_RUNTIME_TAG}
  )

  FetchContent_GetProperties(unified-runtime)
  FetchContent_MakeAvailable(unified-runtime)

  set(UNIFIED_RUNTIME_SOURCE_DIR
    "${unified-runtime_SOURCE_DIR}" CACHE PATH
    "Path to Unified Runtime Headers" FORCE)
elseif(LIBSYCL_UR_SOURCE_DIR)
  # LIBSYCL_UR_USE_FETCH_CONTENT is OFF and LIBSYCL_UR_SOURCE_DIR has been set,
  # use the external Unified Runtime source directory.
  set(UNIFIED_RUNTIME_SOURCE_DIR
    "${LIBSYCL_UR_SOURCE_DIR}" CACHE PATH
    "Path to Unified Runtime Headers" FORCE)
  add_subdirectory(
    ${UNIFIED_RUNTIME_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/unified-runtime)
else()
  message(FATAL_ERROR
    "LIBSYCL_UR_USE_FETCH_CONTENT is disabled but no alternative Unified \
    Runtime source directory has been provided, either:

    * Set -DLIBSYCL_UR_SOURCE_DIR=/path/to/unified-runtime
    * Clone the UR repo in ${CMAKE_CURRENT_SOURCE_DIR}/unified-runtime")
endif()

# Restore original flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

message(STATUS
  "Using Unified Runtime source directory: ${UNIFIED_RUNTIME_SOURCE_DIR}")

set(UNIFIED_RUNTIME_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/include")
set(UNIFIED_RUNTIME_SRC_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source")
set(UNIFIED_RUNTIME_COMMON_INCLUDE_DIR "${UNIFIED_RUNTIME_SOURCE_DIR}/source/common")

add_library(UnifiedRuntimeLoader ALIAS ur_loader)
add_library(UnifiedRuntimeCommon ALIAS ur_common)
add_library(UnifiedMemoryFramework ALIAS ur_umf)

add_library(UnifiedRuntime-Headers INTERFACE)

target_include_directories(UnifiedRuntime-Headers
  INTERFACE
    "${UNIFIED_RUNTIME_INCLUDE_DIR}"
)

find_package(Threads REQUIRED)

if(TARGET UnifiedRuntimeLoader)
  # Install the UR loader.
  install(TARGETS ur_loader
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-runtime-loader
    ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-runtime-loader
    RUNTIME DESTINATION "bin" COMPONENT unified-runtime-loader
  )
endif()

add_custom_target(UnifiedRuntimeAdapters)

function(add_sycl_ur_adapter NAME)
  add_dependencies(UnifiedRuntimeAdapters ur_adapter_${NAME})

  install(TARGETS ur_adapter_${NAME}
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT ur_adapter_${NAME}
    RUNTIME DESTINATION "bin" COMPONENT ur_adapter_${NAME})

  set(manifest_file
    ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_ur_adapter_${NAME}.txt)
  add_custom_command(OUTPUT ${manifest_file}
    COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_INSTALL_COMPONENT=ur_adapter_${NAME}"
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    COMMENT "Deploying component ur_adapter_${NAME}"
    USES_TERMINAL
  )
  add_custom_target(install-sycl-ur-adapter-${NAME}
    DEPENDS ${manifest_file} ur_adapter_${NAME}
  )
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

install(TARGETS umf
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-memory-framework
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT unified-memory-framework
  RUNTIME DESTINATION "bin" COMPONENT unified-memory-framework)
