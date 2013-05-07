# See www/CMake.html for instructions on how to build libcxx with CMake.

#===============================================================================
# Setup Project
#===============================================================================

project(libcxx CXX C)
cmake_minimum_required(VERSION 2.8)

set(PACKAGE_NAME libcxx)
set(PACKAGE_VERSION trunk-svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

#===============================================================================
# Setup CMake Options
#===============================================================================

# Define options.
option(LIBCXX_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXX_ENABLE_RTTI "Use run time type information." ON)
option(LIBCXX_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXX_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXX_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXX_ENABLE_CXX0X "Enable -std=c++0x and use of c++0x language features if the compiler supports it." ON)
option(LIBCXX_ENABLE_SHARED "Build libc++ as a shared library." ON)

set(CXXABIS none libcxxabi libcxxrt libsupc++)
if (NOT DEFINED LIBCXX_CXX_ABI)
  set(LIBCXX_CXX_ABI "none")
endif()
set(LIBCXX_CXX_ABI "${LIBCXX_CXX_ABI}" CACHE STRING
    "Specify C++ ABI library to use." FORCE)
set_property(CACHE LIBCXX_CXX_ABI PROPERTY STRINGS "";${CXXABIS})

#===============================================================================
# Configure System
#===============================================================================

# Get triples.
include(GetTriple)
get_host_triple(LIBCXX_HOST_TRIPLE
  LIBCXX_HOST_ARCH
  LIBCXX_HOST_VENDOR
  LIBCXX_HOST_OS
  )
set(LIBCXX_HOST_TRIPLE ${LIBCXX_HOST_TRIPLE} CACHE STRING "Host triple.")
get_target_triple(LIBCXX_TARGET_TRIPLE
  LIBCXX_TARGET_ARCH
  LIBCXX_TARGET_VENDOR
  LIBCXX_TARGET_OS
  )
set(LIBCXX_TARGET_TRIPLE ${LIBCXX_TARGET_TRIPLE} CACHE STRING "Target triple.")

#===============================================================================
# Add an ABI library if appropriate
#===============================================================================

#
# _setup_abi: Set up the build to use an ABI library
#
# Parameters:
#   abidefines: A list of defines needed to compile libc++ with the ABI library
#   abilibs   : A list of libraries to link against
#   abifiles  : A list of files (which may be relative paths) to copy into the
#               libc++ build tree for the build.  These files will also be
#               installed alongside the libc++ headers.
#   abidirs   : A list of relative paths to create under an include directory
#               in the libc++ build directory.
#
macro(setup_abi_lib abipathvar abidefines abilibs abifiles abidirs)
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS ${abidefines})
  set(${abipathvar} "${${abipathvar}}"
    CACHE STRINGS
    "Paths to ABI include directories separate by ';'."
    )
  set(LIBCXX_CXX_ABI_LIBRARIES ${abilibs})
  set(LIBCXX_ABILIB_FILES ${abifiles})
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
  foreach(_d ${abidirs})
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/${_d}")
  endforeach()
  set(LIBCXX_ABILIB_FILE_PATHS)
  foreach(fpath ${LIBCXX_ABILIB_FILES})
    set(found FALSE)
    foreach(incpath ${${abipathvar}})
      if (EXISTS "${incpath}/${fpath}")
        set(found TRUE)
        get_filename_component(dstdir ${fpath} PATH)
        get_filename_component(ifile ${fpath} NAME)
        add_custom_command(
          OUTPUT "${CMAKE_BINARY_DIR}/include/${dstdir}/${ifile}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${incpath}/${fpath}"
            "${CMAKE_BINARY_DIR}/include/${dstdir}"
          MAIN_DEPENDENCY "${incpath}/${fpath}"
          )
        list(APPEND LIBCXX_CXX_ABI_DEPS
          "${CMAKE_BINARY_DIR}/include/${dstdir}/${ifile}"
          )
      endif()
    endforeach()
    if (NOT found)
      message(FATAL_ERROR "Failed to find ${fpath}")
    endif()
  endforeach()
  add_custom_target(abilib_headers DEPENDS ${LIBCXX_CXX_ABI_DEPS})
  set(LIBCXX_CXX_ABI_DEPS abilib_headers)
  include_directories("${CMAKE_BINARY_DIR}/include")
  install(DIRECTORY "${CMAKE_BINARY_DIR}/include/"
    DESTINATION include/c++/v1
    FILES_MATCHING
    PATTERN "*"
    )
endmacro()

if ("${LIBCXX_CXX_ABI}" STREQUAL "libsupc++")
  set(_LIBSUPCXX_INCLUDE_FILES
    cxxabi.h bits/c++config.h bits/os_defines.h bits/cpu_defines.h
    bits/cxxabi_tweaks.h bits/cxxabi_forced.h
    )
  setup_abi_lib("LIBCXX_LIBSUPCXX_INCLUDE_PATHS" "-D__GLIBCXX__"
    "supc++" "${_LIBSUPCXX_INCLUDE_FILES}" "bits"
    )
elseif ("${LIBCXX_CXX_ABI}" STREQUAL "libcxxabi")
  setup_abi_lib("LIBCXX_LIBCXXABI_INCLUDE_PATHS" ""
    "c++abi" "cxxabi.h;cxa_demangle.h" ""
    )
elseif ("${LIBCXX_CXX_ABI}" STREQUAL "libcxxrt")
  setup_abi_lib("LIBCXX_LIBCXXRT_INCLUDE_PATHS" "-DLIBCXXRT"
    "cxxrt" "cxxabi.h;unwind.h;unwind-arm.h;unwind-itanium.h" ""
    )
elseif (NOT "${LIBCXX_CXX_ABI}" STREQUAL "none")
  message(FATAL_ERROR
    "Currently libsupc++, libcxxabi, libcxxrt and none are "
    "supported for c++ abi."
    )
endif ()

# Configure compiler.
include(config-ix)

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Get required flags.
# On all systems the system c++ standard library headers need to be excluded.
if (MSVC)
  # MSVC only has -X, which disables all default includes; including the crt.
  # Thus, we do nothing and hope we don't accidentally include any of the C++
  # headers.
else()
  if (LIBCXX_HAS_NOSTDINCXX_FLAG)
    set(LIBCXX_CXX_REQUIRED_FLAGS -nostdinc++)
  endif()
  if (LIBCXX_ENABLE_CXX0X AND LIBCXX_HAS_STDCXX0X_FLAG)
    list(APPEND LIBCXX_CXX_REQUIRED_FLAGS -std=c++0x)
  endif()
endif()

macro(append_if list condition var)
  if (${condition})
    list(APPEND ${list} ${var})
  endif()
endmacro()

# Get warning flags
append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WALL_FLAG -Wall)
append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_W_FLAG -W)
append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WNO_UNUSED_PARAMETER_FLAG -Wno-unused-parameter)
append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WWRITE_STRINGS_FLAG -Wwrite-strings)
append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WNO_LONG_LONG_FLAG -Wno-long-long)
if (LIBCXX_ENABLE_WERROR)
  append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WERROR_FLAG -Werror)
  append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_WX_FLAG -WX)
endif()
if (LIBCXX_ENABLE_PEDANTIC)
  append_if(LIBCXX_CXX_WARNING_FLAGS LIBCXX_HAS_PEDANTIC_FLAG -pedantic)
endif()

# Get feature flags.
# Exceptions
if (LIBCXX_ENABLE_EXCEPTIONS)
  # Catches C++ exceptions only and tells the compiler to assume that extern C
  # functions never throw a C++ exception.
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_EHSC_FLAG -EHsc)
else()
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_LIBCPP_NO_EXCEPTIONS)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_EHS_FLAG -EHs-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_EHA_FLAG -EHa-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions)
endif()
# RTTI
if (NOT LIBCXX_ENABLE_RTTI)
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_LIBCPP_NO_RTTI)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_GR_FLAG -GR-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_FNO_RTTI_FLAG -fno-rtti)
endif()
# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBCXX_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -DNDEBUG)
  endif()
endif()
# Static library
if (NOT LIBCXX_ENABLE_SHARED)
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_LIBCPP_BUILD_STATIC)
endif()

# This is the _ONLY_ place where add_definitions is called.
add_definitions(
 ${LIBCXX_CXX_REQUIRED_FLAGS}
 ${LIBCXX_CXX_WARNING_FLAGS}
 ${LIBCXX_CXX_FEATURE_FLAGS}
 )

#===============================================================================
# Setup Source Code
#===============================================================================

include_directories(include)

# Add source code. This also contains all of the logic for deciding linker flags
# soname, etc...
add_subdirectory(lib)

#===============================================================================
# Setup Tests
#===============================================================================

add_subdirectory(test)
