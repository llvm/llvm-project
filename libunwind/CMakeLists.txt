#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.4.3)

if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Set MACOSX_RPATH=YES by default
endif()

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR LIBUNWIND_STANDALONE_BUILD)
  project(libunwind LANGUAGES C CXX ASM)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  if(NOT LLVM_CONFIG_PATH)
    find_program(LLVM_CONFIG_PATH "llvm-config")
  endif()
  if (DEFINED LLVM_PATH)
    set(LLVM_INCLUDE_DIR ${LLVM_INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_PATH ${LLVM_PATH} CACHE PATH "Path to LLVM source tree")
    set(LLVM_MAIN_SRC_DIR ${LLVM_PATH})
    set(LLVM_CMAKE_PATH "${LLVM_PATH}/cmake/modules")
  elseif(LLVM_CONFIG_PATH)
    message(STATUS "Found LLVM_CONFIG_PATH as ${LLVM_CONFIG_PATH}")
    set(CONFIG_COMMAND ${LLVM_CONFIG_PATH} "--includedir" "--prefix" "--src-root")
    execute_process(COMMAND ${CONFIG_COMMAND}
                    RESULT_VARIABLE HAD_ERROR
                    OUTPUT_VARIABLE CONFIG_OUTPUT)
    if (NOT HAD_ERROR)
      string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";"
             CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()

    list(GET CONFIG_OUTPUT 0 INCLUDE_DIR)
    list(GET CONFIG_OUTPUT 1 LLVM_OBJ_ROOT)
    list(GET CONFIG_OUTPUT 2 MAIN_SRC_DIR)

    set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
    set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
    set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
    set(LLVM_LIT_PATH "${LLVM_PATH}/utils/lit/lit.py")

    # --cmakedir is supported since llvm r291218 (4.0 release)
    execute_process(
      COMMAND ${LLVM_CONFIG_PATH} --cmakedir
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
      ERROR_QUIET)
    if(NOT HAD_ERROR)
      string(STRIP "${CONFIG_OUTPUT}" LLVM_CMAKE_PATH_FROM_LLVM_CONFIG)
      file(TO_CMAKE_PATH "${LLVM_CMAKE_PATH_FROM_LLVM_CONFIG}" LLVM_CMAKE_PATH)
    else()
      file(TO_CMAKE_PATH "${LLVM_BINARY_DIR}" LLVM_BINARY_DIR_CMAKE_STYLE)
      set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR_CMAKE_STYLE}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
    endif()
  else()
    message(WARNING "UNSUPPORTED LIBUNWIND CONFIGURATION DETECTED: "
                    "llvm-config not found and LLVM_MAIN_SRC_DIR not defined. "
                    "Reconfigure with -DLLVM_CONFIG=path/to/llvm-config "
                    "or -DLLVM_PATH=path/to/llvm-source-root.")
  endif()

  if (EXISTS ${LLVM_CMAKE_PATH})
    # Enable warnings, otherwise -w gets added to the cflags by HandleLLVMOptions.
    set(LLVM_ENABLE_WARNINGS ON)
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include("${LLVM_CMAKE_PATH}/AddLLVM.cmake")
    include("${LLVM_CMAKE_PATH}/HandleLLVMOptions.cmake")
  else()
    message(WARNING "Not found: ${LLVM_CMAKE_PATH}")
  endif()

  set(PACKAGE_NAME libunwind)
  set(PACKAGE_VERSION 12.0.0git)
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
    set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
  else()
    # Seek installed Lit.
    find_program(LLVM_LIT "lit.py" ${LLVM_MAIN_SRC_DIR}/utils/lit
                 DOC "Path to lit.py")
  endif()

  if (LLVM_LIT)
    # Define the default arguments to use with 'lit', and an option for the user
    # to override.
    set(LIT_ARGS_DEFAULT "-sv")
    if (MSVC OR XCODE)
      set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
    endif()
    set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

    # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
    if (WIN32 AND NOT CYGWIN)
      set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
    endif()
  else()
    set(LLVM_INCLUDE_TESTS OFF)
  endif()

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
else()
  set(LLVM_LIT "${CMAKE_SOURCE_DIR}/utils/lit/lit.py")
endif()

#===============================================================================
# Setup CMake Options
#===============================================================================
include(CMakeDependentOption)
include(HandleCompilerRT)

# Define options.
option(LIBUNWIND_BUILD_32_BITS "Build 32 bit libunwind" ${LLVM_BUILD_32_BITS})
option(LIBUNWIND_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBUNWIND_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBUNWIND_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBUNWIND_ENABLE_SHARED "Build libunwind as a shared library." ON)
option(LIBUNWIND_ENABLE_STATIC "Build libunwind as a static library." ON)
option(LIBUNWIND_ENABLE_CROSS_UNWINDING "Enable cross-platform unwinding support." OFF)
option(LIBUNWIND_ENABLE_ARM_WMMX "Enable unwinding support for ARM WMMX registers." OFF)
option(LIBUNWIND_ENABLE_THREADS "Build libunwind with threading support." ON)
option(LIBUNWIND_WEAK_PTHREAD_LIB "Use weak references to refer to pthread functions." OFF)
option(LIBUNWIND_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)
option(LIBUNWIND_INCLUDE_DOCS "Build the libunwind documentation." ${LLVM_INCLUDE_DOCS})

set(LIBUNWIND_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}" CACHE STRING
    "Define suffix of library directory name (32/64)")
option(LIBUNWIND_INSTALL_LIBRARY "Install the libunwind library." ON)
cmake_dependent_option(LIBUNWIND_INSTALL_STATIC_LIBRARY
  "Install the static libunwind library." ON
  "LIBUNWIND_ENABLE_STATIC;LIBUNWIND_INSTALL_LIBRARY" OFF)
cmake_dependent_option(LIBUNWIND_INSTALL_SHARED_LIBRARY
  "Install the shared libunwind library." ON
  "LIBUNWIND_ENABLE_SHARED;LIBUNWIND_INSTALL_LIBRARY" OFF)
set(LIBUNWIND_TARGET_TRIPLE "" CACHE STRING "Target triple for cross compiling.")
set(LIBUNWIND_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain for cross compiling.")
set(LIBUNWIND_SYSROOT "" CACHE PATH "Sysroot for cross compiling.")
set(LIBUNWIND_TEST_LINKER_FLAGS "" CACHE STRING
    "Additional linker flags for test programs.")
set(LIBUNWIND_TEST_COMPILER_FLAGS "" CACHE STRING
    "Additional compiler flags for test programs.")
set(LIBUNWIND_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.in" CACHE STRING
    "The Lit testing configuration to use when running the tests.")

if (NOT LIBUNWIND_ENABLE_SHARED AND NOT LIBUNWIND_ENABLE_STATIC)
  message(FATAL_ERROR "libunwind must be built as either a shared or static library.")
endif()

# Check that we can build with 32 bits if requested.
if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32)
  if (LIBUNWIND_BUILD_32_BITS AND NOT LLVM_BUILD_32_BITS) # Don't duplicate the output from LLVM
    message(STATUS "Building 32 bits executables and libraries.")
  endif()
elseif(LIBUNWIND_BUILD_32_BITS)
  message(FATAL_ERROR "LIBUNWIND_BUILD_32_BITS=ON is not supported on this platform.")
endif()

option(LIBUNWIND_HERMETIC_STATIC_LIBRARY
  "Do not export any symbols from the static library." OFF)

#===============================================================================
# Configure System
#===============================================================================

# Add path for custom modules
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})

set(LIBUNWIND_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBUNWIND_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}/c++)
  set(LIBUNWIND_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE}/c++)
  if(LIBCXX_LIBDIR_SUBDIR)
    string(APPEND LIBUNWIND_LIBRARY_DIR /${LIBUNWIND_LIBDIR_SUBDIR})
    string(APPEND LIBUNWIND_INSTALL_LIBRARY_DIR /${LIBUNWIND_LIBDIR_SUBDIR})
  endif()
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set(LIBUNWIND_INSTALL_LIBRARY_DIR lib${LIBUNWIND_LIBDIR_SUFFIX})
else()
  set(LIBUNWIND_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBUNWIND_LIBDIR_SUFFIX})
  set(LIBUNWIND_INSTALL_LIBRARY_DIR lib${LIBUNWIND_LIBDIR_SUFFIX})
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})

set(LIBUNWIND_INSTALL_PREFIX "" CACHE STRING "Define libunwind destination prefix.")

set(LIBUNWIND_C_FLAGS "")
set(LIBUNWIND_CXX_FLAGS "")
set(LIBUNWIND_COMPILE_FLAGS "")
set(LIBUNWIND_LINK_FLAGS "")

# Include macros for adding and removing libunwind flags.
include(HandleLibunwindFlags)

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Get required flags.
add_target_flags_if(LIBUNWIND_BUILD_32_BITS "-m32")

if(LIBUNWIND_TARGET_TRIPLE)
  add_target_flags("--target=${LIBUNWIND_TARGET_TRIPLE}")
elseif(CMAKE_CXX_COMPILER_TARGET)
  set(LIBUNWIND_TARGET_TRIPLE "${CMAKE_CXX_COMPILER_TARGET}")
endif()
if(LIBUNWIND_GCC_TOOLCHAIN)
  add_target_flags("--gcc-toolchain=${LIBUNWIND_GCC_TOOLCHAIN}")
elseif(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
  set(LIBUNWIND_GCC_TOOLCHAIN "${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
endif()
if(LIBUNWIND_SYSROOT)
  add_target_flags("--sysroot=${LIBUNWIND_SYSROOT}")
elseif(CMAKE_SYSROOT)
  set(LIBUNWIND_SYSROOT "${CMAKE_SYSROOT}")
endif()

if (LIBUNWIND_TARGET_TRIPLE)
  set(TARGET_TRIPLE "${LIBUNWIND_TARGET_TRIPLE}")
endif()

# Configure compiler.
include(config-ix)

if (LIBUNWIND_USE_COMPILER_RT AND NOT LIBUNWIND_HAS_NODEFAULTLIBS_FLAG)
  list(APPEND LIBUNWIND_LINK_FLAGS "-rtlib=compiler-rt")
endif()

add_compile_flags_if_supported(-Werror=return-type)

# Get warning flags
add_compile_flags_if_supported(-W)
add_compile_flags_if_supported(-Wall)
add_compile_flags_if_supported(-Wchar-subscripts)
add_compile_flags_if_supported(-Wconversion)
add_compile_flags_if_supported(-Wmismatched-tags)
add_compile_flags_if_supported(-Wmissing-braces)
add_compile_flags_if_supported(-Wnewline-eof)
add_compile_flags_if_supported(-Wno-unused-function)
add_compile_flags_if_supported(-Wshadow)
add_compile_flags_if_supported(-Wshorten-64-to-32)
add_compile_flags_if_supported(-Wsign-compare)
add_compile_flags_if_supported(-Wsign-conversion)
add_compile_flags_if_supported(-Wstrict-aliasing=2)
add_compile_flags_if_supported(-Wstrict-overflow=4)
add_compile_flags_if_supported(-Wunused-parameter)
add_compile_flags_if_supported(-Wunused-variable)
add_compile_flags_if_supported(-Wwrite-strings)
add_compile_flags_if_supported(-Wundef)

if (LIBUNWIND_ENABLE_WERROR)
  add_compile_flags_if_supported(-Werror)
  add_compile_flags_if_supported(-WX)
else()
  add_compile_flags_if_supported(-Wno-error)
  add_compile_flags_if_supported(-WX-)
endif()

if (LIBUNWIND_ENABLE_PEDANTIC)
  add_compile_flags_if_supported(-pedantic)
endif()

# Get feature flags.
# Exceptions
# Catches C++ exceptions only and tells the compiler to assume that extern C
# functions never throw a C++ exception.
add_cxx_compile_flags_if_supported(-fstrict-aliasing)
add_cxx_compile_flags_if_supported(-EHsc)

# Don't run the linker in this CMake check.
#
# The reason why this was added is that when building libunwind for
# ARM Linux, we need to pass the -funwind-tables flag in order for it to
# work properly with ARM EHABI.
#
# However, when performing CMake checks, adding this flag causes the check
# to produce a false negative, because the compiler generates calls
# to __aeabi_unwind_cpp_pr0, which is defined in libunwind itself,
# which isn't built yet, so the linker complains about undefined symbols.
#
# This leads to libunwind not being built with this flag, which makes
# libunwind quite useless in this setup.
set(_previous_CMAKE_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
add_compile_flags_if_supported(-funwind-tables)
set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_previous_CMAKE_TRY_COMPILE_TARGET_TYPE})

if (LIBUNWIND_USES_ARM_EHABI AND NOT LIBUNWIND_SUPPORTS_FUNWIND_TABLES_FLAG)
  message(SEND_ERROR "The -funwind-tables flag must be supported "
                     "because this target uses ARM Exception Handling ABI")
endif()

add_cxx_compile_flags_if_supported(-fno-exceptions)
add_cxx_compile_flags_if_supported(-fno-rtti)

# Ensure that we don't depend on C++ standard library.
if (LIBUNWIND_HAS_NOSTDINCXX_FLAG)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -nostdinc++)
  # Remove -stdlib flags to prevent them from causing an unused flag warning.
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBUNWIND_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    add_compile_flags(-D_DEBUG)
  endif()

  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    add_compile_flags(-UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    add_compile_flags(-DNDEBUG)
  endif()
endif()

# Cross-unwinding
if (NOT LIBUNWIND_ENABLE_CROSS_UNWINDING)
  add_compile_flags(-D_LIBUNWIND_IS_NATIVE_ONLY)
endif()

# Threading-support
if (NOT LIBUNWIND_ENABLE_THREADS)
  add_compile_flags(-D_LIBUNWIND_HAS_NO_THREADS)
endif()

# ARM WMMX register support
if (LIBUNWIND_ENABLE_ARM_WMMX)
  # __ARM_WMMX is a compiler pre-define (as per the ACLE 2.0). Clang does not
  # define this macro for any supported target at present. Therefore, here we
  # provide the option to explicitly enable support for WMMX registers in the
  # unwinder.
  add_compile_flags(-D__ARM_WMMX)
endif()

# This is the _ONLY_ place where add_definitions is called.
if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Disable DLL annotations on Windows for static builds.
if (WIN32 AND LIBUNWIND_ENABLE_STATIC AND NOT LIBUNWIND_ENABLE_SHARED)
  add_definitions(-D_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS)
endif()

if (LIBUNWIND_HAS_COMMENT_LIB_PRAGMA)
  if (LIBUNWIND_HAS_DL_LIB)
    add_definitions(-D_LIBUNWIND_LINK_DL_LIB)
  endif()
  if (LIBUNWIND_HAS_PTHREAD_LIB)
    add_definitions(-D_LIBUNWIND_LINK_PTHREAD_LIB)
  endif()
endif()

#===============================================================================
# Setup Source Code
#===============================================================================

include_directories(include)

add_subdirectory(src)

if (LIBUNWIND_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

if (EXISTS ${LLVM_CMAKE_PATH})
  add_subdirectory(test)
endif()
