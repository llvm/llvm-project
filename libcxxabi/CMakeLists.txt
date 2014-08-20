#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 2.8.8)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libcxxabi)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--bindir"
      "--includedir"
      "--libdir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
  else()
    message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
  endif()

  list(GET CONFIG_OUTPUT 0 TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 1 INCLUDE_DIR)
  list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 LLVM_OBJ_ROOT)
  list(GET CONFIG_OUTPUT 4 MAIN_SRC_DIR)

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/share/llvm/cmake")
  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
    include("${LLVM_CMAKE_PATH}/AddLLVM.cmake")
    include("${LLVM_CMAKE_PATH}/HandleLLVMOptions.cmake")
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()

  set(PACKAGE_NAME libcxxabi)
  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

  if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
    set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
  else()
    # Seek installed Lit.
    find_program(LLVM_LIT "lit.py" ${LLVM_MAIN_SRC_DIR}/utils/lit
      DOC "Path to lit.py")
  endif()

  if(LLVM_LIT)
    # Define the default arguments to use with 'lit', and an option for the user
    # to override.
    set(LIT_ARGS_DEFAULT "-sv")
    if (MSVC OR XCODE)
      set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
    endif()
    set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

    # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
    if( WIN32 AND NOT CYGWIN )
      set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
    endif()
  else()
    set(LLVM_INCLUDE_TESTS OFF)
  endif()

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  set(LIBCXXABI_BUILT_STANDALONE 1)
else()
  set(LLVM_LIT "${CMAKE_SOURCE_DIR}/utils/lit/lit.py")
endif()

#===============================================================================
# Setup CMake Options
#===============================================================================

# Define options.
option(LIBCXXABI_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXXABI_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXXABI_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXXABI_USE_LLVM_UNWINDER "Build and use the LLVM unwinder." OFF)

# Default to building a shared library so that the default options still test
# the libc++abi that is being built. There are two problems with testing a
# static libc++abi. In the case of a standalone build, the tests will link the
# system's libc++, which might not have been built against our libc++abi. In the
# case of an in tree build, libc++ will prefer a dynamic libc++abi from the
# system over a static libc++abi from the output directory.
option(LIBCXXABI_ENABLE_SHARED "Build libc++abi as a shared library." ON)

find_path(
  LIBCXXABI_LIBCXX_INCLUDES
  vector
  PATHS ${LIBCXXABI_LIBCXX_INCLUDES}
        ${CMAKE_BINARY_DIR}/${LIBCXXABI_LIBCXX_INCLUDES}
        ${LLVM_MAIN_SRC_DIR}/projects/libcxx/include
        ${LLVM_INCLUDE_DIR}/c++/v1
  )

set(LIBCXXABI_LIBCXX_INCLUDES "${LIBCXXABI_LIBCXX_INCLUDES}" CACHE STRING
    "Specify path to libc++ includes." FORCE)

#===============================================================================
# Configure System
#===============================================================================

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

# Configure compiler.
include(config-ix)

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Get required flags.
macro(append_if list condition var)
  if (${condition})
    list(APPEND ${list} ${var})
  endif()
endmacro()

if (LIBCXXABI_HAS_NOSTDINCXX_FLAG)
  list(APPEND LIBCXXABI_CXX_REQUIRED_FLAGS -nostdinc++)
endif()

append_if(LIBCXXABI_CXX_REQUIRED_FLAGS LIBCXXABI_HAS_WERROR_FLAG -Werror=return-type)

# Get warning flags
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_W_FLAG -W)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WALL_FLAG -Wall)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WCHAR_SUBSCRIPTS_FLAG -Wchar-subscripts)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WCONVERSION_FLAG -Wconversion)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WMISMATCHED_TAGS_FLAG -Wmismatched-tags)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WMISSING_BRACES_FLAG -Wmissing-braces)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WNEWLINE_EOF_FLAG -Wnewline-eof)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WNO_UNUSED_FUNCTION_FLAG -Wno-unused-function)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSHADOW_FLAG -Wshadow)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSHORTEN_64_TO_32_FLAG -Wshorten-64-to-32)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSIGN_COMPARE_FLAG -Wsign-compare)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSIGN_CONVERSION_FLAG -Wsign-conversion)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSTRICT_ALIASING_FLAG -Wstrict-aliasing=2)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WSTRICT_OVERFLOW_FLAG -Wstrict-overflow=4)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WUNUSED_PARAMETER_FLAG -Wunused-parameter)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WUNUSED_VARIABLE_FLAG -Wunused-variable)
append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WWRITE_STRINGS_FLAG -Wwrite-strings)

if (LIBCXXABI_ENABLE_WERROR)
  append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WERROR_FLAG -Werror)
  append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WX_FLAG -WX)
else()
  append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_WNO_ERROR_FLAG -Wno-error)
  append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_NO_WX_FLAG -WX-)
endif()
if (LIBCXXABI_ENABLE_PEDANTIC)
  append_if(LIBCXXABI_CXX_WARNING_FLAGS LIBCXXABI_HAS_PEDANTIC_FLAG -pedantic)
endif()

# Get feature flags.
# Exceptions
# Catches C++ exceptions only and tells the compiler to assume that extern C
# functions never throw a C++ exception.
append_if(LIBCXXABI_CXX_FEATURE_FLAGS LIBCXXABI_HAS_FSTRICT_ALIASING_FLAG -fstrict-aliasing)
append_if(LIBCXXABI_CXX_FEATURE_FLAGS LIBCXXABI_HAS_EHSC_FLAG -EHsc)

# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBCXXABI_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBCXXABI_CXX_FEATURE_FLAGS -D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXXABI_CXX_FEATURE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXXABI_CXX_FEATURE_FLAGS -DNDEBUG)
  endif()
endif()
# Static library
if (NOT LIBCXXABI_ENABLE_SHARED)
  list(APPEND LIBCXXABI_CXX_FEATURE_FLAGS -D_LIBCPP_BUILD_STATIC)
endif()

# This is the _ONLY_ place where add_definitions is called.
if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

string(REPLACE ";" " " LIBCXXABI_CXX_REQUIRED_FLAGS "${LIBCXXABI_CXX_REQUIRED_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBCXXABI_CXX_REQUIRED_FLAGS}")

string(REPLACE ";" " " LIBCXXABI_CXX_WARNING_FLAGS "${LIBCXXABI_CXX_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBCXXABI_CXX_WARNING_FLAGS}")

string(REPLACE ";" " " LIBCXXABI_CXX_FEATURE_FLAGS "${LIBCXXABI_CXX_FEATURE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBCXXABI_CXX_FEATURE_FLAGS}")

#===============================================================================
# Setup Source Code
#===============================================================================

include_directories(include)

# Add source code. This also contains all of the logic for deciding linker flags
# soname, etc...
add_subdirectory(src)

if (LIBCXXABI_USE_LLVM_UNWINDER)
  add_subdirectory(src/Unwind)
endif()

if(NOT LIBCXXABI_ENABLE_SHARED)
  # TODO: Fix the libc++ cmake files so that libc++abi can be statically linked.
  # As it is now, libc++ will prefer linking against a dynamic libc++abi in the
  # system library paths over a static libc++abi in the out directory. This
  # would test the system library rather than the one we just built, which isn't
  # very helpful.
  message(WARNING "The libc++abi tests are currently only valid when "
                  "LIBCXXABI_ENABLE_SHARED is on, no check target will be "
                  "available!")
else()
  add_subdirectory(test)
endif()
