cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(LLVM_RUNTIMES)

find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(WARNING "Python3 not found, using python2 as a fallback")
  find_package(Python2 COMPONENTS Interpreter REQUIRED)
  if(Python2_VERSION VERSION_LESS 2.7)
    message(SEND_ERROR "Python 2.7 or newer is required")
  endif()

  # Treat python2 as python3
  add_executable(Python3::Interpreter IMPORTED)
  set_target_properties(Python3::Interpreter PROPERTIES
    IMPORTED_LOCATION ${Python2_EXECUTABLE})
  set(Python3_EXECUTABLE ${Python2_EXECUTABLE})
endif()

# This needs to be set before we add any Lit target for `add_lit_target` to
# select the right Lit to run the tests.
set(LLVM_LIT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")

# Automatically define a few variables that are normally set globally by LLVM.
# This is to keep some amount of similarity between the global LLVM build and
# this minimal build.
set(LLVM_UMBRELLA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
list(APPEND CMAKE_MODULE_PATH "${LLVM_UMBRELLA_ROOT}/libcxx/cmake/Modules")
include(HandleOutOfTreeLLVM)

# Override the external Lit to make sure we use the one we generate below.
# TODO: We can remove this once we start relying on the in-tree version of Lit
#       in HandleOutOfTreeLLVM.
set(LLVM_DEFAULT_EXTERNAL_LIT "")

# Include individual runtime projects
set(LLVM_ENABLE_PROJECTS "" CACHE STRING "Semicolon-separated list of runtimes to build.")
foreach(project IN LISTS LLVM_ENABLE_PROJECTS)
  add_subdirectory("${LLVM_UMBRELLA_ROOT}/${project}" "${CMAKE_CURRENT_BINARY_DIR}/${project}")
endforeach()

# Generate the llvm-lit wrapper, needed for testing. This must be done after
# the projects have been added, since the wraper is only generated correctly
# if the test suites have already been added with add_lit_target.
add_subdirectory("${LLVM_UMBRELLA_ROOT}/llvm/utils/llvm-lit" "${CMAKE_CURRENT_BINARY_DIR}/llvm-lit")
