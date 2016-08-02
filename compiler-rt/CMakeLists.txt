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
else()
  set(COMPILER_RT_STANDALONE_BUILD FALSE)
endif()

# The CompilerRT build system requires CMake version 2.8.8 or higher in order
# to use its support for building convenience "libraries" as a collection of
# .o files. This is particularly useful in producing larger, more complex
# runtime libraries.
if (NOT MSVC)
  cmake_minimum_required(VERSION 2.8.8)
else()
  # Version 2.8.12.1 is required to build with Visual Studio 2013.
  cmake_minimum_required(VERSION 2.8.12.1)
endif()

# FIXME: It may be removed when we use 2.8.12.
if(CMAKE_VERSION VERSION_LESS 2.8.12)
  # Invalidate a couple of keywords.
  set(cmake_2_8_12_INTERFACE)
  set(cmake_2_8_12_PRIVATE)
else()
  # Use ${cmake_2_8_12_KEYWORD} intead of KEYWORD in target_link_libraries().
  set(cmake_2_8_12_INTERFACE INTERFACE)
  set(cmake_2_8_12_PRIVATE PRIVATE)
  if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW) # automatic when 2.8.12 is required
  endif()
endif()

# Top level target used to build all compiler-rt libraries.
add_custom_target(compiler-rt ALL)

option(COMPILER_RT_BUILD_BUILTINS "Build builtins" ON)
mark_as_advanced(COMPILER_RT_BUILD_BUILTINS)
option(COMPILER_RT_BUILD_SANITIZERS "Build sanitizers" ON)
mark_as_advanced(COMPILER_RT_BUILD_SANITIZERS)

if (NOT COMPILER_RT_STANDALONE_BUILD)
  # Compute the Clang version from the LLVM version.
  # FIXME: We should be able to reuse CLANG_VERSION variable calculated
  #        in Clang cmake files, instead of copying the rules here.
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
         ${PACKAGE_VERSION})
  # Setup the paths where compiler-rt runtimes and headers should be stored.
  set(COMPILER_RT_OUTPUT_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION})
  set(COMPILER_RT_EXEC_OUTPUT_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  set(COMPILER_RT_INSTALL_PATH lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION})
  option(COMPILER_RT_INCLUDE_TESTS "Generate and build compiler-rt unit tests."
         ${LLVM_INCLUDE_TESTS})
 option(COMPILER_RT_ENABLE_WERROR "Fail and stop if warning is triggered"
        ${LLVM_ENABLE_WERROR})
  # Use just-built Clang to compile/link tests on all platforms, except for
  # Windows where we need to use clang-cl instead.
  if(NOT MSVC)
    set(COMPILER_RT_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)
  else()
    set(COMPILER_RT_TEST_COMPILER ${LLVM_RUNTIME_OUTPUT_INTDIR}/clang.exe)
  endif()
else()
  # Take output dir and install path from the user.
  set(COMPILER_RT_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH
    "Path where built compiler-rt libraries should be stored.")
  set(COMPILER_RT_EXEC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH
    "Path where built compiler-rt executables should be stored.")
  set(COMPILER_RT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX} CACHE PATH
    "Path where built compiler-rt libraries should be installed.")
  option(COMPILER_RT_INCLUDE_TESTS "Generate and build compiler-rt unit tests." OFF)
  option(COMPILER_RT_ENABLE_WERROR "Fail and stop if warning is triggered" OFF)
  # Use a host compiler to compile/link tests.
  set(COMPILER_RT_TEST_COMPILER ${CMAKE_C_COMPILER} CACHE PATH "Compiler to use for testing")

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
  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR_CMAKE_STYLE}/share/llvm/cmake")
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

if("${COMPILER_RT_TEST_COMPILER}" MATCHES "clang[+]*$")
  set(COMPILER_RT_TEST_COMPILER_ID Clang)
elseif("${COMPILER_RT_TEST_COMPILER}" MATCHES "clang.*.exe$")
  set(COMPILER_RT_TEST_COMPILER_ID Clang)
else()
  set(COMPILER_RT_TEST_COMPILER_ID GNU)
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

string(TOLOWER ${CMAKE_SYSTEM_NAME} COMPILER_RT_OS_DIR)
set(COMPILER_RT_LIBRARY_OUTPUT_DIR
  ${COMPILER_RT_OUTPUT_DIR}/lib/${COMPILER_RT_OS_DIR})
set(COMPILER_RT_LIBRARY_INSTALL_DIR
  ${COMPILER_RT_INSTALL_PATH}/lib/${COMPILER_RT_OS_DIR})

# Add path for custom compiler-rt modules.
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )
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
include(CheckIncludeFile)
check_include_file(unwind.h HAVE_UNWIND_H)

include(config-ix)

if(MSVC)
  append_string_if(COMPILER_RT_HAS_W3_FLAG /W3 CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
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
append_list_if(COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer SANITIZER_COMMON_CFLAGS)
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
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      string(REGEX REPLACE "/M[DT]d" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      string(REGEX REPLACE "/D_DEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()
  append_list_if(COMPILER_RT_HAS_Oy_FLAG /Oy- SANITIZER_COMMON_CFLAGS)
  append_list_if(COMPILER_RT_HAS_GS_FLAG /GS- SANITIZER_COMMON_CFLAGS)
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

if(APPLE AND SANITIZER_MIN_OSX_VERSION VERSION_LESS "10.9")
  # Mac OS X prior to 10.9 had problems with exporting symbols from
  # libc++/libc++abi.
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
