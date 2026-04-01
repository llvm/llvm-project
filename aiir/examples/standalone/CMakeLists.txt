cmake_minimum_required(VERSION 3.20.0)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(EXTERNAL_PROJECT_BUILD FALSE)
  find_package(AIIR REQUIRED CONFIG)

  # Define the default argument to use by `lit` when testing.
  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

  message(STATUS "Using AIIRConfig.cmake in: ${AIIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(AIIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${AIIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddAIIR)
  include(HandleLLVMOptions)
else()
  set(EXTERNAL_PROJECT_BUILD TRUE)
  # Build via external projects mechanism
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/tools/Standalone/lib)

  set(AIIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(AIIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../aiir)
  set(AIIR_INCLUDE_DIR ${AIIR_MAIN_SRC_DIR}/include)
  set(AIIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/aiir/include)
  set(AIIR_INCLUDE_DIRS "${AIIR_INCLUDE_DIR};${AIIR_GENERATED_INCLUDE_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${AIIR_MAIN_SRC_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
endif()

if(AIIR_ENABLE_BINDINGS_PYTHON)
  include(AIIRDetectPythonEnv)
  aiir_configure_python_dev_packages()
endif()

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${AIIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(AIIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  include(AIIRDetectPythonEnv)
  aiir_configure_python_dev_packages()
  # Note: for EXTERNAL_PROJECT_BUILD this must be set from the command line.
  if(NOT AIIR_PYTHON_PACKAGE_PREFIX)
    set(AIIR_PYTHON_PACKAGE_PREFIX "aiir_standalone" CACHE STRING "" FORCE)
  endif()
  if(NOT AIIR_BINDINGS_PYTHON_NB_DOMAIN)
    set(AIIR_BINDINGS_PYTHON_NB_DOMAIN "aiir_standalone" CACHE STRING "" FORCE)
  endif()
  if(NOT AIIR_BINDINGS_PYTHON_INSTALL_PREFIX)
    set(AIIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/standalone/${AIIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)
  endif()
  add_subdirectory(python)
endif()
if(AIIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
add_subdirectory(standalone-opt)
if(NOT WIN32)
  add_subdirectory(standalone-plugin)
endif()
add_subdirectory(standalone-translate)
