cmake_minimum_required(VERSION 3.13.4)

# Use old version of target_sources command which converts the source
# file paths to full paths.
cmake_policy(SET CMP0076 OLD)
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# The top-level source directory of libc.
set(LIBC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# The top-level directory in which libc is being built.
set(LIBC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Path libc/scripts directory.
set(LIBC_BUILD_SCRIPTS_DIR "${LIBC_SOURCE_DIR}/utils/build_scripts")


set(LIBC_TARGET_OS ${CMAKE_SYSTEM_NAME})
string(TOLOWER ${LIBC_TARGET_OS} LIBC_TARGET_OS)

set(LIBC_TARGET_MACHINE ${CMAKE_SYSTEM_PROCESSOR})

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

option(LLVM_LIBC_ENABLE_LINTING "Enables linting of libc source files" ON)
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
else()
  message(WARNING "
    Linting for libc is currently disabled.

    This is not recommended, to enable set LLVM_LIBC_ENABLE_LINTING to ON
    (pass -DLLVM_LIBC_ENABLE_LINTING=ON to cmake).")
endif()

include(CMakeParseArguments)
include(LLVMLibCRules)
include(LLVMLibCCheckCpuFeatures)

add_subdirectory(include)
add_subdirectory(config)

include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/entrypoints.txt")
include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/headers.txt")

set(TARGET_ENTRYPOINT_NAME_LIST "")
foreach(entrypoint IN LISTS TARGET_LIBC_ENTRYPOINTS TARGET_LIBM_ENTRYPOINTS)
  string(FIND ${entrypoint} "." last_dot_loc REVERSE)
  if(${last_dot_loc} EQUAL -1)
    message(FATAL "Invalid entrypoint target name ${entrypoint}; Expected a '.' "
                  "(dot) in the name.")
  endif()
  math(EXPR name_loc "${last_dot_loc} + 1")
  string(SUBSTRING ${entrypoint} ${name_loc} -1 entrypoint_name)
  list(APPEND TARGET_ENTRYPOINT_NAME_LIST ${entrypoint_name})
endforeach()

add_subdirectory(src)
add_subdirectory(utils)

# The loader can potentially depend on the library components so add it
# after the library implementation directories.
add_subdirectory(loader)

# The lib and test directories are added at the very end as tests
# and libraries potentially draw from the components present in all
# of the other directories.
add_subdirectory(lib)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(fuzzing)
endif()

add_subdirectory(benchmarks)
