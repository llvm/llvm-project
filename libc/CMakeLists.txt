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
