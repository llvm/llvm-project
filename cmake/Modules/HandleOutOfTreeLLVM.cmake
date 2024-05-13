if (NOT DEFINED LLVM_PATH)
  set(LLVM_PATH ${CMAKE_CURRENT_LIST_DIR}/../../llvm CACHE PATH "" FORCE)
endif()

if(NOT IS_DIRECTORY ${LLVM_PATH})
  message(FATAL_ERROR
    "The provided LLVM_PATH (${LLVM_PATH}) is not a valid directory. Note that "
    "building libc++ outside of the monorepo is not supported anymore. Please "
    "use a Standalone build against the monorepo, a Runtimes build or a classic "
    "monorepo build.")
endif()

set(LLVM_INCLUDE_DIR ${LLVM_PATH}/include CACHE PATH "Path to llvm/include")
set(LLVM_PATH ${LLVM_PATH} CACHE PATH "Path to LLVM source tree")
set(LLVM_MAIN_SRC_DIR ${LLVM_PATH})
set(LLVM_CMAKE_DIR "${LLVM_PATH}/cmake/modules")
set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_CURRENT_BINARY_DIR}/lib")

if (EXISTS "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
elseif (EXISTS "${LLVM_MAIN_SRC_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
else()
  message(FATAL_ERROR "Neither ${LLVM_CMAKE_DIR} nor ${LLVM_MAIN_SRC_DIR}/cmake/modules found. "
                      "This is not a supported configuration.")
endif()

message(STATUS "Configuring for standalone build.")

# By default, we target the host, but this can be overridden at CMake invocation time.
include(GetHostTriple)
get_host_triple(LLVM_INFERRED_HOST_TRIPLE)
set(LLVM_HOST_TRIPLE "${LLVM_INFERRED_HOST_TRIPLE}" CACHE STRING "Host on which LLVM binaries will run")
set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING "Target triple used by default.")

# Add LLVM Functions --------------------------------------------------------
if (WIN32)
  set(LLVM_ON_UNIX 0)
  set(LLVM_ON_WIN32 1)
else()
  set(LLVM_ON_UNIX 1)
  set(LLVM_ON_WIN32 0)
endif()

include(AddLLVM OPTIONAL)

# LLVM Options --------------------------------------------------------------
if (NOT DEFINED LLVM_INCLUDE_TESTS)
  set(LLVM_INCLUDE_TESTS ON)
endif()
if (NOT DEFINED LLVM_INCLUDE_DOCS)
  set(LLVM_INCLUDE_DOCS ON)
endif()
if (NOT DEFINED LLVM_ENABLE_SPHINX)
  set(LLVM_ENABLE_SPHINX OFF)
endif()

if (LLVM_INCLUDE_TESTS)
  # Required LIT Configuration ------------------------------------------------
  # Define the default arguments to use with 'lit', and an option for the user
  # to override.
  set(LLVM_DEFAULT_EXTERNAL_LIT "${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py")
  set(LIT_ARGS_DEFAULT "-sv --show-xfail --show-unsupported")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
endif()

# Required doc configuration
if (LLVM_ENABLE_SPHINX)
  find_package(Sphinx REQUIRED)
endif()

if (LLVM_ON_UNIX AND NOT APPLE)
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 1)
else()
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
endif()
