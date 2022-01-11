include(ExternalProject)

set(BOLT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BOLT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 14)

set(BOLT_ENABLE_RUNTIME OFF)
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(BOLT_ENABLE_RUNTIME ON)
endif()

set(BOLT_INCLUDE_TESTS OFF)
if (LLVM_INCLUDE_TESTS)
  string(FIND "${LLVM_ENABLE_PROJECTS}" "clang" POSITION)
  if (NOT ${POSITION} EQUAL -1)
    string(FIND "${LLVM_ENABLE_PROJECTS}" "lld" POSITION)
    if (NOT ${POSITION} EQUAL -1)
      set(BOLT_INCLUDE_TESTS ON)
    else()
      message(WARNING "Not including BOLT tests since lld is disabled")
    endif()
  else()
    message(WARNING "Not including BOLT tests since clang is disabled")
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

# Get the current git revision for BOLT.
find_program(git_executable NAMES git git.exe git.cmd)
if (git_executable)
  execute_process(COMMAND ${git_executable} rev-parse HEAD
    WORKING_DIRECTORY ${LLVM_MAIN_SRC_DIR}
    TIMEOUT 5
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output)
  if( git_result EQUAL 0 )
    string(STRIP "${git_output}" git_ref_id)
    set(BOLT_REVISION "${git_ref_id}")
  endif()
endif()

# If we can't find a revision, set it to "<unknown>".
if (NOT BOLT_REVISION)
  set(BOLT_REVISION "<unknown>")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bolt/Utils/BoltRevision.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/bolt/Utils/BoltRevision.inc)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

add_subdirectory(lib)
add_subdirectory(tools)

if (BOLT_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

option(BOLT_INCLUDE_DOCS "Generate build targets for the BOLT docs."
       ${LLVM_INCLUDE_DOCS})
if (BOLT_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
