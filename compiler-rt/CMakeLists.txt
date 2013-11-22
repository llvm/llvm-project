# CMake build for CompilerRT.
#
# This build assumes that CompilerRT is checked out into the
# 'projects/compiler-rt' inside of an LLVM tree, it is not a stand-alone build
# system.
#
# An important constraint of the build is that it only produces libraries
# based on the ability of the host toolchain to target various platforms.

include(LLVMParseArguments)

# The CompilerRT build system requires CMake version 2.8.8 or higher in order
# to use its support for building convenience "libraries" as a collection of
# .o files. This is particularly useful in producing larger, more complex
# runtime libraries.
cmake_minimum_required(VERSION 2.8.8)

# Top level target used to build all compiler-rt libraries.
add_custom_target(compiler-rt)

# Compute the Clang version from the LLVM version.
# FIXME: We should be able to reuse CLANG_VERSION variable calculated
#        in Clang cmake files, instead of copying the rules here.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
       ${PACKAGE_VERSION})
# Setup the paths where compiler-rt runtimes and headers should be stored.
set(LIBCLANG_INSTALL_PATH lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION})
string(TOLOWER ${CMAKE_SYSTEM_NAME} LIBCLANG_OS_DIR)
set(CLANG_RESOURCE_DIR ${LLVM_BINARY_DIR}/lib/clang/${CLANG_VERSION})
set(COMPILER_RT_LIBRARY_OUTPUT_DIR ${CLANG_RESOURCE_DIR}/lib/${LIBCLANG_OS_DIR})
set(COMPILER_RT_LIBRARY_INSTALL_DIR
  ${LIBCLANG_INSTALL_PATH}/lib/${LIBCLANG_OS_DIR})

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )
include(AddCompilerRT)

set(COMPILER_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COMPILER_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
# Setup custom SDK sysroots.
set(COMPILER_RT_DARWIN_SDK_SYSROOT ${COMPILER_RT_SOURCE_DIR}/SDKs/darwin)
set(COMPILER_RT_LINUX_SDK_SYSROOT ${COMPILER_RT_SOURCE_DIR}/SDKs/linux)
include(SanitizerUtils)

# Detect whether the current target platform is 32-bit or 64-bit, and setup
# the correct commandline flags needed to attempt to target 32-bit and 64-bit.
if (NOT CMAKE_SIZEOF_VOID_P EQUAL 4 AND
    NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Please use architecture with 4 or 8 byte pointers.")
endif()
if (NOT MSVC)
  set(TARGET_64_BIT_CFLAGS "-m64")
  set(TARGET_32_BIT_CFLAGS "-m32")
else()
  set(TARGET_64_BIT_CFLAGS "")
  set(TARGET_32_BIT_CFLAGS "")
endif()

# List of architectures we can target.
set(COMPILER_RT_SUPPORTED_ARCH)

function(get_target_flags_for_arch arch out_var)
  list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
  if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Unsupported architecture: ${arch}")
  else()
    set(${out_var} ${TARGET_${arch}_CFLAGS} PARENT_SCOPE)
  endif()
endfunction()

# Try to compile a very simple source file to ensure we can target the given
# platform. We use the results of these tests to build only the various target
# runtime libraries supported by our current compilers cross-compiling
# abilities.
set(SIMPLE_SOURCE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/simple.c)
file(WRITE ${SIMPLE_SOURCE} "#include <stdlib.h>\nint main() {}")

# test_target_arch(<arch> <target flags...>)
# Sets the target flags for a given architecture and determines if this
# architecture is supported by trying to build a simple file.
macro(test_target_arch arch)
  set(TARGET_${arch}_CFLAGS ${ARGN})
  try_compile(CAN_TARGET_${arch} ${CMAKE_BINARY_DIR} ${SIMPLE_SOURCE}
              COMPILE_DEFINITIONS "${TARGET_${arch}_CFLAGS}"
              CMAKE_FLAGS "-DCMAKE_EXE_LINKER_FLAGS:STRING=${TARGET_${arch}_CFLAGS}")
  if(${CAN_TARGET_${arch}})
    list(APPEND COMPILER_RT_SUPPORTED_ARCH ${arch})
  endif()
endmacro()

if("${LLVM_NATIVE_ARCH}" STREQUAL "X86")
  if (NOT MSVC)
    test_target_arch(x86_64 ${TARGET_64_BIT_CFLAGS})
  endif()
  test_target_arch(i386 ${TARGET_32_BIT_CFLAGS})
elseif("${LLVM_NATIVE_ARCH}" STREQUAL "PowerPC")
  test_target_arch(powerpc64 ${TARGET_64_BIT_CFLAGS})
endif()

# We only support running instrumented tests when we're not cross compiling
# and target a unix-like system. On Android we define the rules for building
# unit tests, but don't execute them.
if("${CMAKE_HOST_SYSTEM}" STREQUAL "${CMAKE_SYSTEM}" AND UNIX AND NOT ANDROID)
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" ON)
else()
  option(COMPILER_RT_CAN_EXECUTE_TESTS "Can we execute instrumented tests" OFF)
endif()

# Check if compiler-rt is built with libc++.
find_flag_in_string("${CMAKE_CXX_FLAGS}" "-stdlib=libc++"
                    COMPILER_RT_USES_LIBCXX)

function(filter_available_targets out_var)
  set(archs)
  foreach(arch ${ARGN})
    list(FIND COMPILER_RT_SUPPORTED_ARCH ${arch} ARCH_INDEX)
    if(NOT (ARCH_INDEX EQUAL -1) AND CAN_TARGET_${arch})
      list(APPEND archs ${arch})
    endif()
  endforeach()
  set(${out_var} ${archs} PARENT_SCOPE)
endfunction()

option(COMPILER_RT_DEBUG "Build runtimes with full debug info" OFF)

# COMPILER_RT_DEBUG_PYBOOL is used by lit.common.configured.in.
pythonize_bool(COMPILER_RT_DEBUG)

# Provide some common commmandline flags for Sanitizer runtimes.
if (NOT MSVC)
  set(SANITIZER_COMMON_CFLAGS
    -fPIC
    -fno-builtin
    -fno-exceptions
    -fomit-frame-pointer
    -funwind-tables
    -fno-stack-protector
    -Wno-gnu  # Variadic macros with 0 arguments for ...
    -fvisibility=hidden
    )
  if (NOT COMPILER_RT_DEBUG)
    list(APPEND SANITIZER_COMMON_CFLAGS -O3)
  endif()
else()
  set(SANITIZER_COMMON_CFLAGS
    /MT
    /Zi
    /Oy-
    /GS-
    /wd4722
    )
endif()
# Build sanitizer runtimes with debug info. (MSVC gets /Zi above)
if (NOT MSVC)
  check_cxx_compiler_flag(-gline-tables-only SUPPORTS_GLINE_TABLES_ONLY_FLAG)
  if(SUPPORTS_GLINE_TABLES_ONLY_FLAG AND NOT COMPILER_RT_DEBUG)
    list(APPEND SANITIZER_COMMON_CFLAGS -gline-tables-only)
  else()
    list(APPEND SANITIZER_COMMON_CFLAGS -g)
  endif()
endif()
# Warnings suppressions.
check_cxx_compiler_flag(-Wno-variadic-macros SUPPORTS_NO_VARIADIC_MACROS_FLAG)
if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-variadic-macros)
endif()
check_cxx_compiler_flag(-Wno-c99-extensions SUPPORTS_NO_C99_EXTENSIONS_FLAG)
if(SUPPORTS_NO_C99_EXTENSIONS_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-c99-extensions)
endif()
# Sanitizer may not have libstdc++, so we can have problems with virtual
# destructors.
check_cxx_compiler_flag(-Wno-non-virtual-dtor SUPPORTS_NO_NON_VIRTUAL_DTOR_FLAG)
if (SUPPORTS_NO_NON_VIRTUAL_DTOR_FLAG)
  list(APPEND SANITIZER_COMMON_CFLAGS -Wno-non-virtual-dtor)
endif()
check_cxx_compiler_flag(-Wglobal-constructors SUPPORTS_GLOBAL_CONSTRUCTORS_FLAG)
# Not all sanitizers forbid global constructors.

if(APPLE)
  # Obtain the iOS Simulator SDK path from xcodebuild.
  execute_process(
    COMMAND xcodebuild -version -sdk iphonesimulator Path
    OUTPUT_VARIABLE IOSSIM_SDK_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(SANITIZER_COMMON_SUPPORTED_DARWIN_OS osx)
  if (IOSSIM_SDK_DIR)
    list(APPEND SANITIZER_COMMON_SUPPORTED_DARWIN_OS iossim)
  endif()

  if(COMPILER_RT_USES_LIBCXX)
    set(SANITIZER_MIN_OSX_VERSION 10.7)
  else()
    set(SANITIZER_MIN_OSX_VERSION 10.6)
  endif()
  set(DARWIN_osx_CFLAGS -mmacosx-version-min=${SANITIZER_MIN_OSX_VERSION})
  set(DARWIN_iossim_CFLAGS 
    -mios-simulator-version-min=7.0 -isysroot ${IOSSIM_SDK_DIR})
  set(DARWIN_osx_LINKFLAGS)
  set(DARWIN_iossim_LINKFLAGS
    -Wl,-ios_simulator_version_min,7.0.0
    -mios-simulator-version-min=7.0
    -isysroot ${IOSSIM_SDK_DIR})
endif()

# Architectures supported by Sanitizer runtimes. Specific sanitizers may
# support only subset of these (e.g. TSan works on x86_64 only).
filter_available_targets(SANITIZER_COMMON_SUPPORTED_ARCH
  x86_64 i386 powerpc64)

add_subdirectory(include)

set(SANITIZER_COMMON_LIT_TEST_DEPS
  clang clang-headers FileCheck count not llvm-nm llvm-symbolizer
  compiler-rt-headers)
# Check code style when running lit tests for sanitizers.
if(UNIX)
  list(APPEND SANITIZER_COMMON_LIT_TEST_DEPS SanitizerLintCheck)
endif()

add_subdirectory(lib)

if(LLVM_INCLUDE_TESTS)
  # Currently the tests have not been ported to CMake, so disable this
  # directory.
  #
  #add_subdirectory(test)
endif()
