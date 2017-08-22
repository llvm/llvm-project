# CMake build for CompilerRT.
#
# This build assumes that CompilerRT is checked out into the
# 'projects/compiler-rt' or 'runtimes/compiler-rt' inside of an LLVM tree.
# Standalone build system for CompilerRT is not yet ready.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

cmake_minimum_required(VERSION 3.4.3)

# Check if compiler-rt is built as a standalone project.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR COMPILER_RT_STANDALONE_BUILD)
  project(CompilerRT C CXX ASM)
  set(COMPILER_RT_STANDALONE_BUILD TRUE)
endif()

# Add path for custom compiler-rt modules.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CFG_RESOLVED_INTDIR "${CMAKE_CFG_INTDIR}/")
else()
  set(CMAKE_CFG_RESOLVED_INTDIR "")
endif()

include(base-config-ix)
include(CompilerRTUtils)

option(COMPILER_RT_BUILD_BUILTINS "Build builtins" ON)
mark_as_advanced(COMPILER_RT_BUILD_BUILTINS)
option(COMPILER_RT_BUILD_SANITIZERS "Build sanitizers" ON)
mark_as_advanced(COMPILER_RT_BUILD_SANITIZERS)
option(COMPILER_RT_BUILD_XRAY "Build xray" ON)
mark_as_advanced(COMPILER_RT_BUILD_XRAY)
option(COMPILER_RT_BUILD_LIBFUZZER "Build libFuzzer" ON)
mark_as_advanced(COMPILER_RT_BUILD_LIBFUZZER)
option(COMPILER_RT_BUILD_XRAY_NO_PREINIT "Build xray with no preinit patching" OFF)
mark_as_advanced(COMPILER_RT_BUILD_XRAY_NO_PREINIT)

set(COMPILER_RT_BAREMETAL_BUILD OFF CACHE BOOLEAN
  "Build for a bare-metal target.")

if (COMPILER_RT_STANDALONE_BUILD)
  load_llvm_config()

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

construct_compiler_rt_default_triple()
if ("${COMPILER_RT_DEFAULT_TARGET_ABI}" STREQUAL "androideabi")
  set(ANDROID 1)
endif()

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

include(HandleCompilerRT)
include(config-ix)

if(APPLE AND SANITIZER_MIN_OSX_VERSION AND SANITIZER_MIN_OSX_VERSION VERSION_LESS "10.9")
  # Mac OS X prior to 10.9 had problems with exporting symbols from
  # libc++/libc++abi.
  set(use_cxxabi_default OFF)
elseif(MSVC)
  set(use_cxxabi_default OFF)
else()
  set(use_cxxabi_default ON)
endif()

option(SANITIZER_CAN_USE_CXXABI "Sanitizers can use cxxabi" ${use_cxxabi_default})
pythonize_bool(SANITIZER_CAN_USE_CXXABI)

set(SANITIZER_CXX_ABI "default" CACHE STRING
    "Specify C++ ABI library to use.")
set(CXXABIS none default libcxxabi libstdc++ libc++)
set_property(CACHE SANITIZER_CXX_ABI PROPERTY STRINGS ;${CXXABIS})

if (SANITIZER_CXX_ABI STREQUAL "default")
  if (HAVE_LIBCXXABI AND COMPILER_RT_DEFAULT_TARGET_ONLY)
    set(SANITIZER_CXX_ABI_LIBNAME "libcxxabi")
    set(SANITIZER_CXX_ABI_INTREE 1)
  elseif (APPLE)
    set(SANITIZER_CXX_ABI_LIBNAME "libcxxabi")
    set(SANITIZER_CXX_ABI_SYSTEM 1)
  else()
    set(SANITIZER_CXX_ABI_LIBNAME "libstdc++")
  endif()
else()
  set(SANITIZER_CXX_ABI_LIBNAME "${SANITIZER_CXX_ABI}")
endif()

if (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libcxxabi")
  if (SANITIZER_CXX_ABI_INTREE)
    if (TARGET unwind_shared OR HAVE_LIBUNWIND)
      list(APPEND SANITIZER_CXX_ABI_LIBRARY unwind_shared)
    endif()
    if (TARGET cxxabi_shared OR HAVE_LIBCXXABI)
      list(APPEND SANITIZER_CXX_ABI_LIBRARY cxxabi_shared)
    endif()
  else()
    list(APPEND SANITIZER_CXX_ABI_LIBRARY "c++abi")
  endif()
elseif (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libc++")
  list(APPEND SANITIZER_CXX_ABI_LIBRARY "c++")
elseif (SANITIZER_CXX_ABI_LIBNAME STREQUAL "libstdc++")
  append_list_if(COMPILER_RT_HAS_LIBSTDCXX stdc++ SANITIZER_CXX_ABI_LIBRARY)
endif()

option(SANITIZER_USE_COMPILER_RT "Use compiler-rt builtins instead of libgcc" OFF)

#================================
# Setup Compiler Flags
#================================

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
if(NOT WIN32)
  append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions SANITIZER_COMMON_CFLAGS)
if(NOT COMPILER_RT_DEBUG AND NOT APPLE)
  append_list_if(COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG -fomit-frame-pointer SANITIZER_COMMON_CFLAGS)
endif()
append_list_if(COMPILER_RT_HAS_FUNWIND_TABLES_FLAG -funwind-tables SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_STACK_PROTECTOR_FLAG -fno-stack-protector SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_SANITIZE_SAFE_STACK_FLAG -fno-sanitize=safe-stack SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_HIDDEN_FLAG -fvisibility=hidden SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FVISIBILITY_INLINES_HIDDEN_FLAG -fvisibility-inlines-hidden SANITIZER_COMMON_CFLAGS)
append_list_if(COMPILER_RT_HAS_FNO_LTO_FLAG -fno-lto SANITIZER_COMMON_CFLAGS)

# The following is a workaround for powerpc64le. This is the only architecture
# that requires -fno-function-sections to work properly. If lacking, the ASan
# Linux test function-sections-are-bad.cc fails with the following error:
# 'undefined symbol: __sanitizer_unaligned_load32'.
if(DEFINED TARGET_powerpc64le_CFLAGS)
  append_list_if(COMPILER_RT_HAS_FNO_FUNCTION_SECTIONS_FLAG -fno-function-sections TARGET_powerpc64le_CFLAGS)
endif()

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
  if (MSVC_VERSION GREATER 1899 OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND SANITIZER_COMMON_CFLAGS /Zc:threadSafeInit-)
  endif()
endif()

append_list_if(COMPILER_RT_DEBUG -DSANITIZER_DEBUG=1 SANITIZER_COMMON_CFLAGS)

# If we're using MSVC,
# always respect the optimization flags set by CMAKE_BUILD_TYPE instead.
if (NOT MSVC)

  # Build with optimization, unless we're in debug mode. 
  if(COMPILER_RT_DEBUG)
    list(APPEND SANITIZER_COMMON_CFLAGS -O0)
  else()
    list(APPEND SANITIZER_COMMON_CFLAGS -O3)
  endif()
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
elseif(MSVC)
  # Use /Z7 instead of /Zi for the asan runtime. This avoids the LNK4099
  # warning from the MS linker complaining that it can't find the 'vc140.pdb'
  # file used by our object library compilations.
  list(APPEND SANITIZER_COMMON_CFLAGS /Z7)
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "/Z[i7I]" "/Z7")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_DEBUG "/Z[i7I]" "/Z7")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Z[i7I]" "/Z7")
endif()

if(LLVM_ENABLE_MODULES)
  # Sanitizers cannot be built with -fmodules. The interceptors intentionally
  # don't include system headers, which is incompatible with modules.
  list(APPEND SANITIZER_COMMON_CFLAGS -fno-modules)
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

# Set common link flags.
append_list_if(COMPILER_RT_HAS_NODEFAULTLIBS_FLAG -nodefaultlibs SANITIZER_COMMON_LINK_FLAGS)

if (SANITIZER_USE_COMPILER_RT)
  list(APPEND SANITIZER_COMMON_LINK_FLAGS -rtlib=compiler-rt)
  find_compiler_rt_library(builtins COMPILER_RT_BUILTINS_LIBRARY)
  list(APPEND SANITIZER_COMMON_LINK_LIBS ${COMPILER_RT_BUILTINS_LIBRARY})
else()
  if (ANDROID)
    append_list_if(COMPILER_RT_HAS_GCC_LIB gcc SANITIZER_COMMON_LINK_LIBS)
  else()
    append_list_if(COMPILER_RT_HAS_GCC_S_LIB gcc_s SANITIZER_COMMON_LINK_LIBS)
  endif()
endif()

append_list_if(COMPILER_RT_HAS_LIBC c SANITIZER_COMMON_LINK_LIBS)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Fuchsia")
  list(APPEND SANITIZER_COMMON_LINK_FLAGS -Wl,-z,defs,-z,now,-z,relro)
  list(APPEND SANITIZER_COMMON_LINK_LIBS magenta)
endif()

# Warnings to turn off for all libraries, not just sanitizers.
append_string_if(COMPILER_RT_HAS_WUNUSED_PARAMETER_FLAG -Wno-unused-parameter CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

if (CMAKE_LINKER MATCHES "link.exe$")
  # Silence MSVC linker warnings caused by empty object files. The
  # sanitizer libraries intentionally use ifdefs that result in empty
  # files, rather than skipping these files in the build system.
  # Ideally, we would pass this flag only for the libraries that need
  # it, but CMake doesn't seem to have a way to set linker flags for
  # individual static libraries, so we enable the suppression flag for
  # the whole compiler-rt project.
  append("/IGNORE:4221" CMAKE_STATIC_LINKER_FLAGS)
endif()

add_subdirectory(include)

set(COMPILER_RT_LIBCXX_PATH ${LLVM_MAIN_SRC_DIR}/projects/libcxx)
if(EXISTS ${COMPILER_RT_LIBCXX_PATH}/)
  set(COMPILER_RT_HAS_LIBCXX_SOURCES TRUE)
else()
  set(COMPILER_RT_LIBCXX_PATH ${LLVM_MAIN_SRC_DIR}/../libcxx)
  if(EXISTS ${COMPILER_RT_LIBCXX_PATH}/)
    set(COMPILER_RT_HAS_LIBCXX_SOURCES TRUE)
  else()
    set(COMPILER_RT_HAS_LIBCXX_SOURCES FALSE)
  endif()
endif()

set(COMPILER_RT_LLD_PATH ${LLVM_MAIN_SRC_DIR}/tools/lld)
if(EXISTS ${COMPILER_RT_LLD_PATH}/ AND LLVM_TOOL_LLD_BUILD)
  set(COMPILER_RT_HAS_LLD TRUE)
else()
  set(COMPILER_RT_LLD_PATH ${LLVM_MAIN_SRC_DIR}/../lld)
  if(EXISTS ${COMPILER_RT_LLD_PATH}/ AND LLVM_TOOL_LLD_BUILD)
    set(COMPILER_RT_HAS_LLD TRUE)
  else()
    set(COMPILER_RT_HAS_LLD FALSE)
  endif()
endif()
pythonize_bool(COMPILER_RT_HAS_LLD)

add_subdirectory(lib)

if(COMPILER_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()
