# CMake build for CompilerRT.
#
# This build assumes that CompilerRT is checked out into the
# 'projects/compiler-rt' inside of an LLVM tree.
# Standalone build system for CompilerRT is not yet ready.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

# Check if compiler-rt is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(CompilerRT C CXX ASM)
  set(COMPILER_RT_STANDALONE_BUILD TRUE)
endif()

cmake_minimum_required(VERSION 3.4.3)
# FIXME:
# The OLD behavior (pre 3.2) for this policy is to not set the value of the 
# CMAKE_EXE_LINKER_FLAGS variable in the generated test project. The NEW behavior
# for this policy is to set the value of the CMAKE_EXE_LINKER_FLAGS variable 
# in the test project to the same as it is in the calling project. The new 
# behavior cause the compiler_rt test to fail during try_compile: see
# projects/compiler-rt/cmake/Modules/CompilerRTUtils.cmake:121 such that
# CAN_TARGET_${arch} is not set properly. This results in COMPILER_RT_SUPPORTED_ARCH
# not being updated properly leading to poblems.
cmake_policy(SET CMP0056 OLD)

# Add path for custom compiler-rt modules.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )

include(base-config-ix)

option(COMPILER_RT_BUILD_BUILTINS "Build builtins" ON)
mark_as_advanced(COMPILER_RT_BUILD_BUILTINS)
option(COMPILER_RT_BUILD_SANITIZERS "Build sanitizers" ON)
mark_as_advanced(COMPILER_RT_BUILD_SANITIZERS)

if (COMPILER_RT_STANDALONE_BUILD)
  if (NOT LLVM_CONFIG_PATH)
    find_program(LLVM_CONFIG_PATH "llvm-config"
                 DOC "Path to llvm-config binary")
    if (NOT LLVM_CONFIG_PATH)
      message(FATAL_ERROR "llvm-config not found: specify LLVM_CONFIG_PATH")
    endif()
  endif()
  execute_process(
    COMMAND ${LLVM_CONFIG_PATH} "--obj-root" "--bindir" "--libdir" "--src-root"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE CONFIG_OUTPUT)
  if (HAD_ERROR)
    message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
  endif()
  string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})
  list(GET CONFIG_OUTPUT 0 LLVM_BINARY_DIR)
  list(GET CONFIG_OUTPUT 1 LLVM_TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LLVM_LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 LLVM_MAIN_SRC_DIR)

  # Make use of LLVM CMake modules.
  file(TO_CMAKE_PATH ${LLVM_BINARY_DIR} LLVM_BINARY_DIR_CMAKE_STYLE)
  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR_CMAKE_STYLE}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
  # Get some LLVM variables from LLVMConfig.
  include("${LLVM_CMAKE_PATH}/LLVMConfig.cmake")

  set(LLVM_LIBRARY_OUTPUT_INTDIR
    ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  # Find Python interpreter.
  set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)
  include(FindPythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "
      Unable to find Python interpreter required testing. Please install Python
      or specify the PYTHON_EXECUTABLE CMake variable.")
  endif()

  # Define default arguments to lit.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")
endif()

set(COMPILER_RT_DEFAULT_TARGET_TRIPLE ${TARGET_TRIPLE} CACHE STRING
    "Default triple for which compiler-rt runtimes will be built.")
if(DEFINED COMPILER_RT_TEST_TARGET_TRIPLE)
  # Backwards compatibility: this variable used to be called
  # COMPILER_RT_TEST_TARGET_TRIPLE.
  set(COMPILER_RT_DEFAULT_TARGET_TRIPLE ${COMPILER_RT_TEST_TARGET_TRIPLE})
endif()

string(REPLACE "-" ";" TARGET_TRIPLE_LIST ${COMPILER_RT_DEFAULT_TARGET_TRIPLE})
list(GET TARGET_TRIPLE_LIST 0 COMPILER_RT_DEFAULT_TARGET_ARCH)
list(GET TARGET_TRIPLE_LIST 1 COMPILER_RT_DEFAULT_TARGET_OS)
list(GET TARGET_TRIPLE_LIST 2 COMPILER_RT_DEFAULT_TARGET_ABI)
# Determine if test target triple is specified explicitly, and doesn't match the
# default.
if(NOT COMPILER_RT_DEFAULT_TARGET_TRIPLE STREQUAL TARGET_TRIPLE)
  set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE TRUE)
else()
  set(COMPILER_RT_HAS_EXPLICIT_DEFAULT_TARGET_TRIPLE FALSE)
endif()
if ("${COMPILER_RT_DEFAULT_TARGET_ABI}" STREQUAL "androideabi")
  set(ANDROID 1)
endif()
include(CompilerRTUtils)

set(COMPILER_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# We support running instrumented tests when we're not cross compiling
# and target a UNIX-like system or Windows.
# We can run tests on Android even when we are cross-compiling.
if(("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND (UNIX OR WIN32)) OR ANDROID
   OR COMPILER_RT_EMULATOR)
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" ON)
else()
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" OFF)
endif()

option(COMPILER_RT_DEBUG "Build runtimes with full debug info" OFF)
option(COMPILER_RT_EXTERNALIZE_DEBUGINFO
  "Generate dSYM files and strip executables and libraries (Darwin Only)" OFF)
# COMPILER_RT_DEBUG_PYBOOL is used by lit.common.configured.in.
pythonize_bool(COMPILER_RT_DEBUG)

#================================
# Setup Compiler Flags
#================================

include(config-ix)

if(MSVC)
  # Override any existing /W flags with /W4. This is what LLVM does.  Failing to
  # remove other /W[0-4] flags will result in a warning about overriding a
  # previous flag.
  if (COMPILER_RT_HAS_W4_FLAG)
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    append_string_if(COMPILER_RT_HAS_W4_FLAG /W4 CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
else()
  append_string_if(COMPILER_RT_HAS_WALL_FLAG -Wall CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
if(COMPILER_RT_ENABLE_WERROR)
  append_string_if(COMPILER_RT_HAS_WERROR_FLAG -Werror CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_string_if(COMPILER_RT_HAS_WX_FLAG /WX CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

append_string_if(COMPILER_RT_HAS_STD_CXX11_FLAG -std=c++11 CMAKE_CXX_FLAGS)

# Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP.
if(NOT COMPILER_RT_HAS_FUNC_SYMBOL)
  add_definitions(-D__func__=__FUNCTION__)
endif()

# Provide some common commmandline flags for Sanitizer runtimes.
append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions SANITIZER_COMMON_CFLAGS)
if(NOT COMPILER_RT_DEBUG)
  append_list_if(COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FUNWIND_TABLES_FLAG -funwind-tables SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG -fno-stack-protector SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_SANITIZE_SAFE_STACK_FLAG -fno-sanitize=safe-stack SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_INLINES_HIDDEN_FLAG -fvisibility-inlines-hidden SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG -fno-function-sections SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_LTO_FLAG -fno-lto SANITIZER_COMMON_CFLAGS)

if(MSVC)
  # Replace the /M[DT][d] flags with /MT, and strip any definitions of _DEBUG,
  # which cause definition mismatches at link time.
  # FIXME: In fact, sanitizers should support both /MT and /MD, see PR20214.
  if(COMPILER_RT_HAS_MT_FLAG)
    foreach(flag_var
      CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      string(REGEX REPLACE "/M[DT]d" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/D_DEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()
  append_list_if(COMPILER_RT_HAS_Oy_FLAG /Oy- SANITIZER_COMMON_CFLAGS)
  append_list_if(COMPILER_RT_HAS_GS_FLAG /GS- SANITIZER_COMMON_CFLAGS)
  # VS 2015 (version 1900) added support for thread safe static initialization.
  # However, ASan interceptors run before CRT initialization, which causes the
  # new thread safe code to crash. Disable this feature for now.
  if (MSVC_VERSION GREATER 1899)
    list(APPEND SANITIZER_COMMON_CFLAGS /Zc:threadSafeInit-)
  endif()
endif()

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 SANITIZER_COMMON_CFLAGS)

# Build with optimization, unless we're in debug mode. If we're using MSVC,
# always respect the optimization flags set by CMAKE_BUILD_TYPE instead.
if(NOT COMPILER_RT_DEBUG AND NOT MSVC)
  list(APPEND SANITIZER_COMMON_CFLAGS -O3)
endif()

# Determine if we should restrict stack frame sizes.
# Stack frames on PowerPC and Mips and in debug biuld can be much larger than
# anticipated.
# FIXME: Fix all sanitizers and add -Wframe-larger-than to
# SANITIZER_COMMON_FLAGS
if(COMPILER_RT_HAS_WFRAME_LARGER_THAN_FLAG AND NOT COMPILER_RT_DEBUG
   AND NOT ${COMPILER_RT_DEFAULT_TARGET_ARCH} MATCHES "powerpc|mips")
  set(SANITIZER_LIMIT_FRAME_SIZE TRUE)
else()
  set(SANITIZER_LIMIT_FRAME_SIZE FALSE)
endif()

# Build sanitizer runtimes with debug info.
if(COMPILER_RT_HAS_GLINE_TABLES_ONLY_FLAG AND NOT COMPILER_RT_DEBUG)
  list(APPEND SANITIZER_COMMON_CFLAGS -gline-tables-only)
elseif(COMPILER_RT_HAS_G_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -g)
elseif(COMPILER_RT_HAS_Zi_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS /Zi)
endif()

# Turn off several warnings.
append_list_if(COMPILER_RT_HAS_WGNU_FLAG -Wno-gnu SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WVARIADIC_MACROS_FLAG -Wno-variadic-macros SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WC99_EXTENSIONS_FLAG -Wno-c99-extensions SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WNON_VIRTUAL_DTOR_FLAG -Wno-non-virtual-dtor SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4146_FLAG /wd4146 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4291_FLAG /wd4291 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4391_FLAG /wd4391 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4722_FLAG /wd4722 SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_WD4800_FLAG /wd4800 SANITIZER_COMMON_CFLAGS)

# Warnings to turn off for all libraries, not just sanitizers.
append_string_if(COMPILER_RT_HAS_WUNUSED_PARAMETER_FLAG -Wno-unused-parameter CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

if(APPLE AND SANITIZER_MIN_OSX_VERSION VERSION_LESS "10.9")
  # Mac OS X prior to 10.9 had problems with exporting symbols from
  # libc++/libc++abi.
  set(SANITIZER_CAN_USE_CXXABI FALSE)
elseif(MSVC)
  set(SANITIZER_CAN_USE_CXXABI FALSE)
else()
  set(SANITIZER_CAN_USE_CXXABI TRUE)
endif()
pythonize_bool(SANITIZER_CAN_USE_CXXABI)

add_subdirectory(include)

set(COMPILER_RT_LIBCXX_PATH ${LLVM_MAIN_SRC_DIR}/projects/libcxx)
if(EXISTS ${COMPILER_RT_LIBCXX_PATH}/)
  set(COMPILER_RT_HAS_LIBCXX_SOURCES TRUE)
else()
  set(COMPILER_RT_HAS_LIBCXX_SOURCES FALSE)
endif()

set(COMPILER_RT_LLD_PATH ${LLVM_MAIN_SRC_DIR}/tools/lld)
if(EXISTS ${COMPILER_RT_LLD_PATH}/)
  set(COMPILER_RT_HAS_LLD_SOURCES TRUE)
else()
  set(COMPILER_RT_HAS_LLD_SOURCES FALSE)
endif()
pythonize_bool(COMPILER_RT_HAS_LLD_SOURCES)

add_subdirectory(lib)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()
