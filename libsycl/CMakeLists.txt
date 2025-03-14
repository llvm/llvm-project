#===============================================================================
# Setup Project
#===============================================================================
cmake_minimum_required(VERSION 3.20.0)

set(LLVM_SUBPROJECT_TITLE "libsycl")

set(LIBSYCL_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBSYCL_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#===============================================================================
# Setup CMake Options
#===============================================================================

option(LIBSYCL_ENABLE_WERROR "Treat all warnings as errors in the libsycl project" OFF)
option(LIBSYCL_DISABLE_STL_ASSERTIONS "Disable assertions in STL containers when building libsycl" ON)
option(LIBSYCL_ENABLE_STACK_PRINTING "Enables stack printing on crashes of SYCL applications" OFF)
option(LIBSYCL_DEBUG_SYMBOLS "Builds SYCL runtime libraries with debug symbols" OFF)

#===============================================================================
# Configure System
#===============================================================================

set(LIBSYCL_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE STRING
    "Path where libsycl headers should be installed.")
set(LIBSYCL_INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}" CACHE STRING
    "Path where built libsycl runtime libraries should be installed.")

set(LIBSYCL_SHARED_OUTPUT_NAME "sycl" CACHE STRING "Output name for the shared libsycl runtime library.")

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBSYCL_TARGET_SUBDIR ${LLVM_DEFAULT_TARGET_TRIPLE})
  if(LIBSYCL_LIBDIR_SUBDIR)
    string(APPEND LIBSYCL_TARGET_SUBDIR /${LIBSYCL_LIBDIR_SUBDIR})
  endif()
  set(LIBSYCL_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LIBSYCL_TARGET_SUBDIR})
  set(LIBSYCL_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LIBSYCL_TARGET_SUBDIR} CACHE STRING
      "Path where built libsycl libraries should be installed.")
  unset(LIBSYCL_TARGET_SUBDIR)
else()
  if(LLVM_LIBRARY_OUTPUT_INTDIR)
    set(LIBSYCL_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  else()
    set(LIBSYCL_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBSYCL_LIBDIR_SUFFIX})
  endif()
  set(LIBSYCL_INSTALL_LIBRARY_DIR lib${LIBSYCL_LIBDIR_SUFFIX} CACHE STRING
      "Path where built libsycl libraries should be installed.")
endif()

set(LIBSYCL_INCLUDE_DIR ${LIBSYCL_SOURCE_DIR}/include)
set(LIBSYCL_BUILD_INCLUDE_DIR ${LLVM_BINARY_DIR}/include)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})

# The change in LIBSYCL_MAJOR_VERSION must be accompanied with the same update in
# llvm/clang/lib/Driver/CMakeLists.txt.
#
# See doc/developer/ABIPolicyGuide.md for the meaning when in the middle of
# development cycle.
set(LIBSYCL_MAJOR_VERSION 0)
set(LIBSYCL_MINOR_VERSION 1)
set(LIBSYCL_PATCH_VERSION 0)
set(LIBSYCL_VERSION_STRING "${LIBSYCL_MAJOR_VERSION}.${LIBSYCL_MINOR_VERSION}.${LIBSYCL_PATCH_VERSION}")

#===============================================================================
# Setup Compiler Flags
#===============================================================================

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  # Skip asynchronous C++ exceptions catching and assume "extern C" functions
  # never throw C++ exceptions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")

  # Add PDB debug information
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(CheckLinkerFlag)
  check_linker_flag(CXX "LINKER:/DEBUG" LINKER_SUPPORTS_DEBUG)
  if(LINKER_SUPPORTS_DEBUG)
    # sccache is not compatible with /Zi flag
    if (CMAKE_CXX_COMPILER_LAUNCHER STREQUAL "sccache")
      # CMake may put /Zi by default
      if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
      elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
      elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
      endif()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
    endif()
    add_link_options("LINKER:/DEBUG")

    # Enable unreferenced removal and ICF in Release mode.
    check_linker_flag(CXX "LINKER:/OPT:REF,/OPT:ICF" LINKER_SUPPORTS_OPTS)
    if (LINKER_SUPPORTS_OPTS AND uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
      add_link_options("LINKER:/OPT:REF" "LINKER:/OPT:ICF")
    endif()
  endif()
endif()

# Enable all warnings by default
if (MSVC)
  set(CMAKE_CXX_FLAGS "/W4 ${CMAKE_CXX_FLAGS}")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
endif()

if(LIBSYCL_ENABLE_WERROR)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "/WX ${CMAKE_CXX_FLAGS}")
    add_definitions(
      -wd4996 # Suppress 'function': was declared deprecated'
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif()

# This is a workaround to detect changes (add or modify) in subtree which
# are not detected by copy_directory command.
# TODO: detect and process remove header/directory case
file(GLOB_RECURSE HEADERS_IN_SYCL_DIR CONFIGURE_DEPENDS "${LIBSYCL_INCLUDE_DIR}/sycl/*")

string(REPLACE "${LIBSYCL_INCLUDE_DIR}" "${LIBSYCL_BUILD_INCLUDE_DIR}"
  OUT_HEADERS_IN_SYCL_DIR "${HEADERS_IN_SYCL_DIR}")

# Copy SYCL headers from sources to build directory
add_custom_target(sycl-headers
DEPENDS ${OUT_HEADERS_IN_SYCL_DIR})

add_custom_command(
  OUTPUT  ${OUT_HEADERS_IN_SYCL_DIR}
  DEPENDS ${HEADERS_IN_SYCL_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBSYCL_INCLUDE_DIR}/sycl ${LIBSYCL_BUILD_INCLUDE_DIR}/sycl
  COMMENT "Copying SYCL headers...")

message(WARNING "${LIBSYCL_INCLUDE_DIR}/sycl ${LIBSYCL_BUILD_INCLUDE_DIR}/sycl")
install(DIRECTORY "${LIBSYCL_INCLUDE_DIR}/sycl" DESTINATION ${LIBSYCL_INSTALL_INCLUDE_DIR} COMPONENT sycl-headers)

if (WIN32)
  set(LIBSYCL_RT_LIBS ${LIBSYCL_SHARED_OUTPUT_NAME}${LIBSYCL_MAJOR_VERSION})
  if (MSVC)
    list(APPEND LIBSYCL_RT_LIBS ${LIBSYCL_SHARED_OUTPUT_NAME}${LIBSYCL_MAJOR_VERSION}d)
  endif()
else()
  set(LIBSYCL_RT_LIBS ${LIBSYCL_SHARED_OUTPUT_NAME})
endif()

if (LLVM_ENABLE_ASSERTIONS AND NOT LIBSYCL_DISABLE_STL_ASSERTIONS AND NOT WIN32)
  add_definitions(-D_GLIBCXX_ASSERTIONS=1)
endif()

add_subdirectory(src)

add_custom_target(libsycl-runtime-libraries
  DEPENDS ${LIBSYCL_RT_LIBS}
)