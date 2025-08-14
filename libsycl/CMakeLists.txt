#===============================================================================
# Setup Project
#===============================================================================
cmake_minimum_required(VERSION 3.20.0)

set(LLVM_SUBPROJECT_TITLE "libsycl")

set(LIBSYCL_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBSYCL_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

#===============================================================================
# Limitations
#===============================================================================

if (CMAKE_SYSTEM_NAME STREQUAL Windows AND NOT MSVC)
# Build with other compilers is not configured, not guaranteed and not tested.
    message(FATAL_ERROR
      "When compiling for Windows, libsycl requires a"
      " version of Microsoft Visual C++ or another compiler"
      " that uses the Visual C++ cl command-line syntax.")
endif()

#===============================================================================
# Setup CMake Options
#===============================================================================

option(LIBSYCL_ENABLE_WERROR "Treat all warnings as errors in the libsycl project" OFF)
option(LIBSYCL_ENABLE_PEDANTIC "Compile with pedantic enabled." OFF)

#===============================================================================
# Configure System
#===============================================================================

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(LIBSYCL_SHARED_OUTPUT_NAME "sycl" CACHE STRING "Output name for the shared libsycl runtime library.")

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBSYCL_TARGET_SUBDIR ${LLVM_DEFAULT_TARGET_TRIPLE})
  if(LIBSYCL_LIBDIR_SUBDIR)
    string(APPEND LIBSYCL_TARGET_SUBDIR /${LIBSYCL_LIBDIR_SUBDIR})
  endif()
  cmake_path(NORMAL_PATH LIBSYCL_TARGET_SUBDIR)
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

set(LIBSYCL_INCLUDE_DIR include)
set(LIBSYCL_BUILD_INCLUDE_DIR ${LLVM_BINARY_DIR}/${LIBSYCL_INCLUDE_DIR})
set(LIBSYCL_SOURCE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBSYCL_LIBRARY_DIR})

set(LIBSYCL_MAJOR_VERSION 0)
set(LIBSYCL_MINOR_VERSION 1)
set(LIBSYCL_PATCH_VERSION 0)
set(LIBSYCL_VERSION_STRING "${LIBSYCL_MAJOR_VERSION}.${LIBSYCL_MINOR_VERSION}.${LIBSYCL_PATCH_VERSION}")
set(LIBSYCL_ABI_NAMESPACE "__V${LIBSYCL_MAJOR_VERSION}" CACHE STRING
    "The inline ABI namespace used by libsycl. It defaults to __Vn where `n` is the current ABI version.")
if (NOT LIBSYCL_ABI_NAMESPACE MATCHES "__V.*")
   message(FATAL_ERROR "LIBSYCL_ABI_NAMESPACE must be a reserved identifier, got '${LIBSYCL_ABI_NAMESPACE}'.")
endif()

#===============================================================================
# Setup build & install rules
#===============================================================================

# Generate headers
configure_file("${LIBSYCL_SOURCE_DIR}/src/version.hpp.in" "${LIBSYCL_BUILD_INCLUDE_DIR}/sycl/__impl/version.hpp")

# Install generated headers.
install(FILES
  "${LIBSYCL_BUILD_INCLUDE_DIR}/sycl/__impl/version.hpp"
  DESTINATION "${LIBSYCL_INCLUDE_DIR}/sycl/__impl"
  COMPONENT sycl-headers)

# This is a workaround to detect changes (add or modify) in subtree which
# are not detected by copy_directory command.
file(GLOB_RECURSE HEADERS_IN_SYCL_DIR CONFIGURE_DEPENDS "${LIBSYCL_SOURCE_INCLUDE_DIR}/sycl/*")
file(GLOB_RECURSE HEADERS_IN_CL_DIR CONFIGURE_DEPENDS "${LIBSYCL_SOURCE_INCLUDE_DIR}/CL/*")

string(REPLACE "${LIBSYCL_SOURCE_INCLUDE_DIR}" "${LIBSYCL_BUILD_INCLUDE_DIR}"
  OUT_HEADERS_IN_SYCL_DIR "${HEADERS_IN_SYCL_DIR}")
string(REPLACE "${LIBSYCL_SOURCE_INCLUDE_DIR}/CL" "${LIBSYCL_BUILD_INCLUDE_DIR}/CL"
  OUT_HEADERS_IN_CL_DIR "${HEADERS_IN_CL_DIR}")

# Copy SYCL headers from sources to build directory
add_custom_target(sycl-headers
  DEPENDS ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR})

add_custom_command(
  OUTPUT  ${OUT_HEADERS_IN_SYCL_DIR}
          ${OUT_HEADERS_IN_CL_DIR}
  DEPENDS ${HEADERS_IN_SYCL_DIR}
          ${HEADERS_IN_CL_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBSYCL_SOURCE_INCLUDE_DIR}/sycl ${LIBSYCL_BUILD_INCLUDE_DIR}/sycl
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBSYCL_SOURCE_INCLUDE_DIR}/CL ${LIBSYCL_BUILD_INCLUDE_DIR}/CL
  COMMENT "Copying SYCL headers...")

install(DIRECTORY "${LIBSYCL_SOURCE_INCLUDE_DIR}/sycl" DESTINATION ${LIBSYCL_INCLUDE_DIR} COMPONENT sycl-headers)
install(DIRECTORY "${LIBSYCL_SOURCE_INCLUDE_DIR}/CL" DESTINATION ${LIBSYCL_INCLUDE_DIR} COMPONENT sycl-headers)

set(LIBSYCL_RT_LIBS ${LIBSYCL_SHARED_OUTPUT_NAME})

add_subdirectory(src)

add_custom_target(libsycl-runtime-libraries
  DEPENDS ${LIBSYCL_RT_LIBS}
)
