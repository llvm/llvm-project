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

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libunwind)

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
  set(PACKAGE_VERSION 9.0.0svn)
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

set(LIBUNWIND_COMPILER    ${CMAKE_CXX_COMPILER})
set(LIBUNWIND_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBUNWIND_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
       ${PACKAGE_VERSION})

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(DEFAULT_INSTALL_PREFIX lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/${LLVM_DEFAULT_TARGET_TRIPLE}/)
  set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/${LLVM_DEFAULT_TARGET_TRIPLE}/lib${LIBUNWIND_LIBDIR_SUFFIX})
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
else()
  set(LIBUNWIND_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBUNWIND_LIBDIR_SUFFIX})
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})

set(LIBUNWIND_INSTALL_PREFIX ${DEFAULT_INSTALL_PREFIX} CACHE STRING
    "Define libunwind destination prefix.")

set(LIBUNWIND_C_FLAGS "")
set(LIBUNWIND_CXX_FLAGS "")
set(LIBUNWIND_COMPILE_FLAGS "")
set(LIBUNWIND_LINK_FLAGS "")

# Get required flags.
macro(unwind_append_if list condition var)
  if (${condition})
    list(APPEND ${list} ${var})
  endif()
endmacro()

macro(add_target_flags_if condition var)
  if (${condition})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${var}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${var}")
    list(APPEND LIBUNWIND_COMPILE_FLAGS ${var})
    list(APPEND LIBUNWIND_LINK_FLAGS ${var})
  endif()
endmacro()

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

#===============================================================================
# Setup Compiler Flags
#===============================================================================

unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WERROR_FLAG -Werror=return-type)

# Get warning flags
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_W_FLAG -W)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WALL_FLAG -Wall)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WCHAR_SUBSCRIPTS_FLAG -Wchar-subscripts)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WCONVERSION_FLAG -Wconversion)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WMISMATCHED_TAGS_FLAG -Wmismatched-tags)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WMISSING_BRACES_FLAG -Wmissing-braces)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WNEWLINE_EOF_FLAG -Wnewline-eof)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WNO_UNUSED_FUNCTION_FLAG -Wno-unused-function)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSHADOW_FLAG -Wshadow)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSHORTEN_64_TO_32_FLAG -Wshorten-64-to-32)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSIGN_COMPARE_FLAG -Wsign-compare)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSIGN_CONVERSION_FLAG -Wsign-conversion)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSTRICT_ALIASING_FLAG -Wstrict-aliasing=2)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WSTRICT_OVERFLOW_FLAG -Wstrict-overflow=4)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WUNUSED_PARAMETER_FLAG -Wunused-parameter)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WUNUSED_VARIABLE_FLAG -Wunused-variable)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WWRITE_STRINGS_FLAG -Wwrite-strings)
unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WUNDEF_FLAG -Wundef)

if (LIBUNWIND_ENABLE_WERROR)
  unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WERROR_FLAG -Werror)
  unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WX_FLAG -WX)
else()
  unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_WNO_ERROR_FLAG -Wno-error)
  unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_NO_WX_FLAG -WX-)
endif()

if (LIBUNWIND_ENABLE_PEDANTIC)
  unwind_append_if(LIBUNWIND_COMPILE_FLAGS LIBUNWIND_HAS_PEDANTIC_FLAG -pedantic)
endif()

# Get feature flags.
# Exceptions
# Catches C++ exceptions only and tells the compiler to assume that extern C
# functions never throw a C++ exception.
unwind_append_if(LIBUNWIND_CXX_FLAGS LIBUNWIND_HAS_FSTRICT_ALIASING_FLAG -fstrict-aliasing)
unwind_append_if(LIBUNWIND_CXX_FLAGS LIBUNWIND_HAS_EHSC_FLAG -EHsc)

unwind_append_if(LIBUNWIND_C_FLAGS LIBUNWIND_HAS_FUNWIND_TABLES -funwind-tables)

# Ensure that we don't depend on C++ standard library.
unwind_append_if(LIBUNWIND_CXX_FLAGS LIBUNWIND_HAS_NOSTDINCXX_FLAG -nostdinc++)

# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBUNWIND_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBUNWIND_COMPILE_FLAGS -D_DEBUG)
  endif()

  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBUNWIND_COMPILE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBUNWIND_COMPILE_FLAGS -DNDEBUG)
  endif()
endif()

# Cross-unwinding
if (NOT LIBUNWIND_ENABLE_CROSS_UNWINDING)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -D_LIBUNWIND_IS_NATIVE_ONLY)
endif()

# Threading-support
if (NOT LIBUNWIND_ENABLE_THREADS)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -D_LIBUNWIND_HAS_NO_THREADS)
endif()

# ARM WMMX register support
if (LIBUNWIND_ENABLE_ARM_WMMX)
  # __ARM_WMMX is a compiler pre-define (as per the ACLE 2.0). Clang does not
  # define this macro for any supported target at present. Therefore, here we
  # provide the option to explicitly enable support for WMMX registers in the
  # unwinder.
  list(APPEND LIBUNWIND_COMPILE_FLAGS -D__ARM_WMMX)
endif()

# This is the _ONLY_ place where add_definitions is called.
if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Disable DLL annotations on Windows for static builds.
if (WIN32 AND LIBUNWIND_ENABLE_STATIC AND NOT LIBUNWIND_ENABLE_SHARED)
  add_definitions(-D_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS)
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
