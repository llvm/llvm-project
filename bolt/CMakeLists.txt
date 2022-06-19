include(ExternalProject)

set(BOLT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BOLT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 14)

set(BOLT_ENABLE_RUNTIME OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(BOLT_ENABLE_RUNTIME ON)
endif()

set(BOLT_CLANG_EXE "" CACHE FILEPATH "Path to clang executable for the target \
architecture for use in BOLT tests")
set(BOLT_LLD_EXE "" CACHE FILEPATH "Path to lld executable for the target \
architecture for use in BOLT tests")

set(BOLT_INCLUDE_TESTS OFF)
if (LLVM_INCLUDE_TESTS)
  set(BOLT_CLANG_PRESENT OFF)
  set(BOLT_LLD_PRESENT OFF)

  if ("clang" IN_LIST LLVM_ENABLE_PROJECTS AND BOLT_CLANG_EXE)
    message(WARNING "BOLT_CLANG_EXE is set and clang project is enabled. \
          BOLT_CLANG_EXE will be used for BOLT tests.")
  endif()
  if ("clang" IN_LIST LLVM_ENABLE_PROJECTS OR BOLT_CLANG_EXE)
    set(BOLT_CLANG_PRESENT ON)
  endif()

  if ("lld" IN_LIST LLVM_ENABLE_PROJECTS AND BOLT_LLD_EXE)
    message(WARNING "BOLT_LLD_EXE is set and lld project is enabled. \
          BOLT_LLD_EXE will be used for BOLT tests.")
  endif()
  if ("lld" IN_LIST LLVM_ENABLE_PROJECTS OR BOLT_LLD_EXE)
    set(BOLT_LLD_PRESENT ON)
  endif()

  if (BOLT_CLANG_PRESENT AND BOLT_LLD_PRESENT)
    set(BOLT_INCLUDE_TESTS ON)
  else()
    message(WARNING "Not including BOLT tests since clang or lld is disabled. \
          Add clang and lld to LLVM_ENABLE_PROJECTS or provide paths to clang \
          and lld binaries in BOLT_CLANG_EXE and BOLT_LLD_EXE.")
  endif()
endif()

if (BOLT_ENABLE_RUNTIME)
  message(STATUS "Building BOLT runtime libraries for X86")
  ExternalProject_Add(bolt_rt
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
    STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-stamps
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-bins
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_BUILD_TYPE=Release
               -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
               -DCMAKE_INSTALL_PREFIX=${LLVM_BINARY_DIR}
    BUILD_ALWAYS True
    )
  install(CODE "execute_process\(COMMAND \${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX} -P ${CMAKE_CURRENT_BINARY_DIR}/bolt_rt-bins/cmake_install.cmake \)"
    COMPONENT bolt)
  add_llvm_install_targets(install-bolt_rt
    DEPENDS bolt_rt bolt
    COMPONENT bolt)
endif()

find_program(GNU_LD_EXECUTABLE NAMES ${LLVM_DEFAULT_TARGET_TRIPLE}-ld.bfd ld.bfd DOC "GNU ld")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

add_subdirectory(lib)
add_subdirectory(tools)

if (BOLT_INCLUDE_TESTS)
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND BOLT_TEST_DEPS BoltUnitTests)
  endif()
  add_subdirectory(test)
endif()

option(BOLT_INCLUDE_DOCS "Generate build targets for the BOLT docs."
       ${LLVM_INCLUDE_DOCS})
if (BOLT_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
