# Check if lld is built as a standalone project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lld)
  cmake_minimum_required(VERSION 3.4.3)

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  set(LLD_BUILT_STANDALONE TRUE)

  find_program(LLVM_CONFIG_PATH "llvm-config" DOC "Path to llvm-config binary")
  if(NOT LLVM_CONFIG_PATH)
    message(FATAL_ERROR "llvm-config not found: specify LLVM_CONFIG_PATH")
  endif()

  execute_process(COMMAND "${LLVM_CONFIG_PATH}" "--obj-root" "--includedir"
                  RESULT_VARIABLE HAD_ERROR
                  OUTPUT_VARIABLE LLVM_CONFIG_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(HAD_ERROR)
    message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
  endif()

  string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" LLVM_CONFIG_OUTPUT "${LLVM_CONFIG_OUTPUT}")

  list(GET LLVM_CONFIG_OUTPUT 0 OBJ_ROOT)
  list(GET LLVM_CONFIG_OUTPUT 1 MAIN_INCLUDE_DIR)

  set(LLVM_OBJ_ROOT ${OBJ_ROOT} CACHE PATH "path to LLVM build tree")
  set(LLVM_MAIN_INCLUDE_DIR ${MAIN_INCLUDE_DIR} CACHE PATH "path to llvm/include")

  file(TO_CMAKE_PATH ${LLVM_OBJ_ROOT} LLVM_BINARY_DIR)
  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")

  if(NOT EXISTS "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
    message(FATAL_ERROR "LLVMConfig.cmake not found")
  endif()
  include("${LLVM_CMAKE_PATH}/LLVMConfig.cmake")

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
  include_directories("${LLVM_BINARY_DIR}/include" ${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
endif()

set(LLD_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLD_INCLUDE_DIR ${LLD_SOURCE_DIR}/include )
set(LLD_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Compute the LLD version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" LLD_VERSION
  ${PACKAGE_VERSION})
message(STATUS "LLD version: ${LLD_VERSION}")

string(REGEX REPLACE "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" "\\1" LLD_VERSION_MAJOR
  ${LLD_VERSION})
string(REGEX REPLACE "[0-9]+\\.([0-9]+)(\\.[0-9]+)?" "\\1" LLD_VERSION_MINOR
  ${LLD_VERSION})

# Determine LLD revision and repository.
# TODO: Figure out a way to get the revision and the repository on windows.
if ( NOT CMAKE_SYSTEM_NAME MATCHES "Windows" )
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetSourceVersion ${LLD_SOURCE_DIR}
                  OUTPUT_VARIABLE LLD_REVISION)

  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/utils/GetRepositoryPath ${LLD_SOURCE_DIR}
                  OUTPUT_VARIABLE LLD_REPOSITORY)
  if ( LLD_REPOSITORY )
    # Replace newline characters with spaces
    string(REGEX REPLACE "(\r?\n)+" " " LLD_REPOSITORY ${LLD_REPOSITORY})
    # Remove leading spaces
    STRING(REGEX REPLACE "^[ \t\r\n]+" "" LLD_REPOSITORY "${LLD_REPOSITORY}" )
    # Remove trailing spaces
    string(REGEX REPLACE "(\ )+$" "" LLD_REPOSITORY ${LLD_REPOSITORY})
  endif()

  if ( LLD_REVISION )
    # Replace newline characters with spaces
    string(REGEX REPLACE "(\r?\n)+" " " LLD_REVISION ${LLD_REVISION})
    # Remove leading spaces
    STRING(REGEX REPLACE "^[ \t\r\n]+" "" LLD_REVISION "${LLD_REVISION}" )
    # Remove trailing spaces
    string(REGEX REPLACE "(\ )+$" "" LLD_REVISION ${LLD_REVISION})
  endif()
endif ()

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/lld/Config/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/lld/Config/Version.inc)


if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

list (APPEND CMAKE_MODULE_PATH "${LLD_SOURCE_DIR}/cmake/modules")

option(LLD_USE_VTUNE
       "Enable VTune user task tracking."
       OFF)
if (LLD_USE_VTUNE)
  find_package(VTune)
  if (VTUNE_FOUND)
    include_directories(${VTune_INCLUDE_DIRS})
    list(APPEND LLVM_COMMON_LIBS ${VTune_LIBRARIES})
    add_definitions(-DLLD_HAS_VTUNE)
  endif()
endif()


if (MSVC)
  add_definitions(-wd4530) # Suppress 'warning C4530: C++ exception handler used, but unwind semantics are not enabled.'
  add_definitions(-wd4062) # Suppress 'warning C4062: enumerator X in switch of enum Y is not handled' from system header.
endif()

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
    )
endif()

macro(add_lld_library name)
  add_llvm_library(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "lld libraries")
endmacro(add_lld_library)


add_subdirectory(lib)
add_subdirectory(tools/lld)

if (LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()

add_subdirectory(docs)
add_subdirectory(COFF)
add_subdirectory(ELF)
