#===--- CMakeLists.txt - Herbception error-domain runtime -------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# libherbceptions — the error-domain runtime used by the 'throws'/'fails{E}'
# herbception exception model. Also a first-class LLVM runtime: add
# "libherbceptions" to LLVM_ENABLE_RUNTIMES.
#
#==========================================================================#

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libherbceptions CXX)
endif()

set(LIBHERBCEPTIONS_VERSION 1.0.0)
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" LIBHERBCEPTIONS_VERSION_MAJOR
  ${LIBHERBCEPTIONS_VERSION})

option(LIBHERBCEPTIONS_BUILD_SHARED "Build the shared library" ON)
option(LIBHERBCEPTIONS_BUILD_STATIC "Build the static library" ON)
option(LIBHERBCEPTIONS_HEADERS_ONLY "Header-only build" OFF)
option(LIBHERBCEPTIONS_FREESTANDING "Build with -ffreestanding" OFF)
option(LIBHERBCEPTIONS_ENABLE_EXCEPTIONS "Enable C++ dynamic exception handling (not herbceptions)" ON)
option(LIBHERBCEPTIONS_ENABLE_RTTI "Enable C++ RTTI" ON)
option(LIBHERBCEPTIONS_ENABLE_EBCDIC "Build with EBCDIC support for guard against -fexec-charset= EBCDIC charsets" OFF)
option(LIBHERBCEPTIONS_BUILD_TESTS "Build the tests" OFF)
option(LIBHERBCEPTIONS_BUILD_FUZZERS "Build the fuzz targets" OFF)
option(LIBHERBCEPTIONS_FUZZ_ENGINE "Link fuzz targets against libFuzzer" OFF)

set(LIBHERBCEPTIONS_SOURCES
  src/posix.cpp
  src/parse.cpp
  src/cmath_errc.cpp
  src/itanium_exception_ptr.cpp
  src/msvc_exception_ptr.cpp
  src/win32.cpp
  src/nt.cpp
  src/com.cpp
  src/wine.cpp
)

add_library(herbceptions_headers INTERFACE)
target_include_directories(herbceptions_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(LIBHERBCEPTIONS_FREESTANDING)
  target_compile_options(herbceptions_headers INTERFACE -ffreestanding)
endif()
if(NOT LIBHERBCEPTIONS_ENABLE_EXCEPTIONS)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(herbceptions_headers INTERFACE -fno-exceptions)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_definitions(herbceptions_headers INTERFACE _HAS_EXCEPTIONS=0)
  endif()
endif()
if(NOT LIBHERBCEPTIONS_ENABLE_RTTI)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(herbceptions_headers INTERFACE -fno-rtti)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_definitions(herbceptions_headers INTERFACE _HAS_RTTI=0)
  endif()
endif()
if(LIBHERBCEPTIONS_ENABLE_EBCDIC)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_definitions(herbceptions_headers INTERFACE __LIBHERBCEPTIONS_ENABLE_EBCDIC=1)
  endif()
endif()

# Headers-only build: no runtime sources.
if(LIBHERBCEPTIONS_HEADERS_ONLY)
  add_library(herbceptions ALIAS herbceptions_headers)
  add_library(herbceptions_static ALIAS herbceptions_headers)
  return()
endif()

if(LIBHERBCEPTIONS_BUILD_SHARED AND NOT LIBHERBCEPTIONS_FREESTANDING)
  add_library(herbceptions SHARED ${LIBHERBCEPTIONS_SOURCES})
endif()
add_library(herbceptions_static STATIC ${LIBHERBCEPTIONS_SOURCES})
if(LIBHERBCEPTIONS_BUILD_SHARED AND NOT LIBHERBCEPTIONS_FREESTANDING)
  set(HERBCEPTIONS_TARGETS herbceptions herbceptions_static)
else()
  set(HERBCEPTIONS_TARGETS herbceptions_static)
endif()
foreach(target IN LISTS HERBCEPTIONS_TARGETS)
  target_link_libraries(${target} PUBLIC herbceptions_headers)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(${target} PRIVATE _HERBCEPTIONS_BUILDING_RUNTIME=1)
  if(LIBHERBCEPTIONS_FREESTANDING)
    target_compile_options(${target} PRIVATE -ffreestanding)
  endif()
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
  )
endforeach()
if(LIBHERBCEPTIONS_BUILD_SHARED AND NOT LIBHERBCEPTIONS_FREESTANDING)
  set_target_properties(herbceptions PROPERTIES
    VERSION ${LIBHERBCEPTIONS_VERSION}
    SOVERSION ${LIBHERBCEPTIONS_VERSION_MAJOR}
  )
endif()

if(TARGET cxx_shared OR TARGET cxx_static OR LLVM_RUNTIMES_BUILD)
  set(LIBHERBCEPTIONS_IN_RUNTIMES ON)
  message(STATUS "libherbceptions runtime hook: LIST_DIR=${CMAKE_CURRENT_LIST_DIR} BIN=${CMAKE_BINARY_DIR}")
  foreach(d "${CMAKE_CURRENT_SOURCE_DIR}/../libcxxabi/include"
            "${CMAKE_BINARY_DIR}/include/c++/v1")
    cmake_path(NORMAL_PATH d)
    if(EXISTS "${d}")
      message(STATUS "libherbceptions inc: ${d}")
    else()
      message(STATUS "libherbceptions missing: ${d}")
    endif()
    if(EXISTS "${d}" AND NOT "${d}" IN_LIST LIBHERBCEPTIONS_LIBCXX_INCLUDES)
      list(APPEND LIBHERBCEPTIONS_LIBCXX_INCLUDES "${d}")
    endif()
  endforeach()
  foreach(target herbceptions herbceptions_static)
    if(LIBHERBCEPTIONS_LIBCXX_INCLUDES)
      target_compile_options(${target} PRIVATE -nostdinc++)
      foreach(d IN LISTS LIBHERBCEPTIONS_LIBCXX_INCLUDES)
        target_compile_options(${target} PRIVATE "-isystem${d}")
      endforeach()
    endif()
    if(TARGET cxx_shared)
      target_link_libraries(${target} PRIVATE cxx_shared)
      add_dependencies(${target} cxx_shared)
    elseif(TARGET cxx_static)
      target_link_libraries(${target} PRIVATE cxx_static)
      add_dependencies(${target} cxx_static)
    endif()
  endforeach()
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND LIBHERBCEPTIONS_BUILD_SHARED)
  set_target_properties(herbceptions_static PROPERTIES
    OUTPUT_NAME herbceptions_static)
else()
  set_target_properties(herbceptions_static PROPERTIES
    OUTPUT_NAME herbceptions)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  if(LIBHERBCEPTIONS_BUILD_SHARED AND NOT LIBHERBCEPTIONS_FREESTANDING)
    target_link_libraries(herbceptions PRIVATE ntdll)
  endif()
  target_link_libraries(herbceptions_static PRIVATE ntdll)
endif()

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN "error"
)
if(LIBHERBCEPTIONS_BUILD_SHARED AND NOT LIBHERBCEPTIONS_FREESTANDING)
  install(TARGETS herbceptions herbceptions_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
else()
  install(TARGETS herbceptions_static
    ARCHIVE DESTINATION lib
  )
endif()

if(LIBHERBCEPTIONS_BUILD_TESTS AND NOT LIBHERBCEPTIONS_HEADERS_ONLY)
  # Lit-based test infrastructure.
  if(NOT TARGET llvm-lit)
    find_program(LLVM_LIT_EXECUTABLE llvm-lit
      HINTS ${LLVM_TOOLS_DIR} $ENV{LLVM_TOOLS_DIR}
      DOC "Path to llvm-lit")
    if(LLVM_LIT_EXECUTABLE)
      message(STATUS "Found llvm-lit: ${LLVM_LIT_EXECUTABLE}")
    else()
      message(STATUS "llvm-lit not found - tests will not be runnable")
    endif()
  endif()

  # Configure the lit site configuration.
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/test/lit.site.cfg.py
    @ONLY)

  # Lit test suite.
  set(LIBHERBCEPTIONS_TEST_DEPENDS)
  if(TARGET herbceptions_static)
    list(APPEND LIBHERBCEPTIONS_TEST_DEPENDS herbceptions_static)
  endif()

  if(LLVM_LIT_EXECUTABLE)
    # Find FileCheck.
    find_program(FILECHECK_EXECUTABLE FileCheck
      HINTS ${LLVM_TOOLS_DIR} $ENV{LLVM_TOOLS_DIR}
      DOC "Path to FileCheck")
    if(FILECHECK_EXECUTABLE)
      message(STATUS "Found FileCheck: ${FILECHECK_EXECUTABLE}")
    else()
      message(STATUS "FileCheck not found - tests may fail")
    endif()

    add_custom_target(check-herbceptions
      COMMAND ${LLVM_LIT_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/test
        --verbose
        --param cxx=${CMAKE_CXX_COMPILER}
        --param filecheck=${FILECHECK_EXECUTABLE}
      DEPENDS ${LIBHERBCEPTIONS_TEST_DEPENDS}
      COMMENT "Running libherbceptions tests"
      USES_TERMINAL)
  endif()

  # Legacy CTest support (kept for compatibility).
  add_executable(herbceptions_domain_test test/domain_test.cpp)
  target_link_libraries(herbceptions_domain_test PRIVATE herbceptions)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(herbceptions_domain_test PRIVATE
      -fherbceptions -fexceptions)
  endif()
  add_test(NAME herbceptions_domain_test COMMAND herbceptions_domain_test)
endif()

if(LIBHERBCEPTIONS_BUILD_FUZZERS AND NOT LIBHERBCEPTIONS_HEADERS_ONLY)
  foreach(target herbceptions herbceptions_static)
    target_compile_options(${target} PRIVATE -fsanitize=fuzzer-no-link)
  endforeach()
  set(LIBHERBCEPTIONS_LIBCXX_DIR
    "$ENV{HOME}/toolchains/llvm/x86_64-linux-gnu/x86_64-linux-gnu/usr"
    CACHE PATH "libc++ installation used by the fuzzer runtime")

  function(add_herbceptions_fuzzer name)
    add_executable(${name} fuzz/${name}.cpp)
    target_include_directories(${name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${name} PRIVATE herbceptions)
    if(LIBHERBCEPTIONS_FUZZ_ENGINE)
      target_compile_definitions(${name} PRIVATE
        LIBHERBCEPTIONS_HAVE_FUZZ_ENGINE=1)
      target_compile_options(${name} PRIVATE -fsanitize=fuzzer
        -nostdinc++
        "-isystem${LIBHERBCEPTIONS_LIBCXX_DIR}/include/c++/v1")
      target_link_options(${name} PRIVATE -fsanitize=fuzzer
        -nostdlib++
        "-L${LIBHERBCEPTIONS_LIBCXX_DIR}/lib" -lc++ -lc++abi)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
      target_link_libraries(${name} PRIVATE ntdll)
    endif()
  endfunction()

  add_herbceptions_fuzzer(error_codes_fuzzer)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      add_herbceptions_fuzzer(msvc_exception_record_fuzzer)
    else()
      add_herbceptions_fuzzer(itanium_exception_record_fuzzer)
    endif()
  else()
    add_herbceptions_fuzzer(itanium_exception_record_fuzzer)
  endif()
endif()
