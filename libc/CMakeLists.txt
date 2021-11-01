cmake_minimum_required(VERSION 3.13.4)

# Use old version of target_sources command which converts the source
# file paths to full paths.
cmake_policy(SET CMP0076 OLD)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# The top-level sourse and binary directories.
set(LIBC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
# The top-level directory in which libc is being built.
set(LIBC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Path libc/scripts directory.
set(LIBC_BUILD_SCRIPTS_DIR "${LIBC_SOURCE_DIR}/utils/build_scripts")

set(LIBC_TARGET_OS ${CMAKE_SYSTEM_NAME})
string(TOLOWER ${LIBC_TARGET_OS} LIBC_TARGET_OS)

# Defines LIBC_TARGET_ARCHITECTURE and associated macros.
include(LLVMLibCArchitectures)
include(LLVMLibCCheckMPFR)

# Flags to pass down to the compiler while building the libc functions.
set(LIBC_COMPILE_OPTIONS_DEFAULT "" CACHE STRING "Architecture to tell clang to optimize for (e.g. -march=... or -mcpu=...)")

# Check --print-resource-dir to find the compiler resource dir if this flag
# is supported by the compiler.
execute_process(
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
  RESULT_VARIABLE COMMAND_RETURN_CODE
  OUTPUT_VARIABLE COMPILER_RESOURCE_DIR
)
# Retrieve the host compiler's resource dir.
if(COMMAND_RETURN_CODE EQUAL 0)
  set(COMPILER_RESOURCE_DIR
    "${COMPILER_RESOURCE_DIR}" CACHE PATH "path to compiler resource dir"
  )
  message(STATUS "Set COMPILER_RESOURCE_DIR to "
                 "${COMPILER_RESOURCE_DIR} using --print-resource-dir")
else()
  set(COMPILER_RESOURCE_DIR OFF)
  message(STATUS "COMPILER_RESOURCE_DIR not set
                  --print-resource-dir not supported by host compiler")
endif()

option(LLVM_LIBC_FULL_BUILD "Build and test LLVM libc as if it is the full libc" OFF)

option(LLVM_LIBC_ENABLE_LINTING "Enables linting of libc source files" OFF)
if(LLVM_LIBC_ENABLE_LINTING AND (NOT LLVM_LIBC_FULL_BUILD))
  message(FATAL_ERROR "Cannot enable linting when full libc build is not enabled.")
endif()
if(LLVM_LIBC_ENABLE_LINTING)
  if("clang-tools-extra" IN_LIST LLVM_ENABLE_PROJECTS
             AND "clang" IN_LIST LLVM_ENABLE_PROJECTS)
    add_custom_target(lint-libc)
  else()
    message(FATAL_ERROR "
      'clang' and 'clang-tools-extra' are required in LLVM_ENABLE_PROJECTS to
      lint llvm-libc. The linting step performs important checks to help prevent
      the introduction of subtle bugs, but it may increase build times.

      To disable linting set LLVM_LIBC_ENABLE_LINTING to OFF
      (pass -DLLVM_LIBC_ENABLE_LINTING=OFF to cmake).")
  endif()
elseif(LLVM_LIBC_FULL_BUILD)
  message(WARNING "
    Linting for libc is currently disabled.

    This is not recommended, to enable set LLVM_LIBC_ENABLE_LINTING to ON
    (pass -DLLVM_LIBC_ENABLE_LINTING=ON to cmake).")
endif()

option(LLVM_LIBC_INCLUDE_SCUDO "Include the SCUDO standalone as the allocator for LLVM libc" OFF)
if(LLVM_LIBC_INCLUDE_SCUDO)
  if (NOT "compiler-rt" IN_LIST LLVM_ENABLE_PROJECTS)
    message(FATAL_ERROR "SCUDO cannot be included without adding compiler-rt to LLVM_ENABLE_PROJECTS")
  endif()
endif()

include(CMakeParseArguments)
include(LLVMLibCRules)
include(LLVMLibCCheckCpuFeatures)

if(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}/entrypoints.txt")
  set(entrypoint_file "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}/entrypoints.txt")
elseif(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/entrypoints.txt")
  set(entrypoint_file "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/entrypoints.txt")
else()
  message(FATAL_ERROR "entrypoints.txt file for the target platform not found.")
endif()
include(${entrypoint_file})

if(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}/headers.txt")
  include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_ARCHITECTURE}/headers.txt")
elseif(EXISTS "${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/headers.txt")
  include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/headers.txt")
endif()

set(TARGET_ENTRYPOINT_NAME_LIST "")
foreach(entrypoint IN LISTS TARGET_LLVMLIBC_ENTRYPOINTS)
  string(FIND ${entrypoint} "." last_dot_loc REVERSE)
  if(${last_dot_loc} EQUAL -1)
    message(FATAL "Invalid entrypoint target name ${entrypoint}; Expected a '.' "
                  "(dot) in the name.")
  endif()
  math(EXPR name_loc "${last_dot_loc} + 1")
  string(SUBSTRING ${entrypoint} ${name_loc} -1 entrypoint_name)
  list(APPEND TARGET_ENTRYPOINT_NAME_LIST ${entrypoint_name})
endforeach()

if(LLVM_LIBC_FULL_BUILD)
  # We need to set up hdrgen first since other targets depend on it.
  add_subdirectory(utils/LibcTableGenUtil)
  add_subdirectory(utils/HdrGen)
endif()

add_subdirectory(include)
add_subdirectory(config)
add_subdirectory(src)
add_subdirectory(utils)

if(LLVM_LIBC_FULL_BUILD)
  # The loader can potentially depend on the library components so add it
  # after the library implementation directories.
  add_subdirectory(loader)
endif()

# The lib and test directories are added at the very end as tests
# and libraries potentially draw from the components present in all
# of the other directories.
add_subdirectory(lib)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(fuzzing)
endif()

add_subdirectory(benchmarks)
