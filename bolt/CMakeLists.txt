cmake_minimum_required(VERSION 3.20.0)

set(LLVM_SUBPROJECT_TITLE "BOLT")

if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(bolt)
  set(BOLT_BUILT_STANDALONE TRUE)
endif()

set(BOLT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BOLT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 17)

# Add path for custom modules.
list(INSERT CMAKE_MODULE_PATH 0 "${BOLT_SOURCE_DIR}/cmake/modules")

include(GNUInstallDirs)

# standalone build, copied from clang
if(BOLT_BUILT_STANDALONE)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
  set(CMAKE_CXX_STANDARD_REQUIRED YES)
  set(CMAKE_CXX_EXTENSIONS NO)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

  set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm" CACHE PATH "Path to LLVM source tree")
  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  # They are used as destination of target generators.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})
  list(APPEND CMAKE_REQUIRED_DEFINITIONS ${LLVM_DEFINITIONS_LIST})

  include(AddLLVM)
  include(TableGen)
  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories("${LLVM_LIBRARY_DIR}")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_INSTALL_BINDIR}" )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/${LLVM_LIBDIR_SUFFIX}" )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/${LLVM_LIBDIR_SUFFIX}")
endif() # standalone

# Determine default set of targets to build -- the intersection of
# those BOLT supports and those LLVM is targeting.
set(BOLT_TARGETS_TO_BUILD_all "AArch64;X86;RISCV")
set(BOLT_TARGETS_TO_BUILD_default)
foreach (tgt ${BOLT_TARGETS_TO_BUILD_all})
  if (tgt IN_LIST LLVM_TARGETS_TO_BUILD)
    list(APPEND BOLT_TARGETS_TO_BUILD_default ${tgt})
  endif()
endforeach()

# Allow the user to specify the BOLT targets, and then check that LLVM
# is indeed targeting those.
set(BOLT_TARGETS_TO_BUILD "${BOLT_TARGETS_TO_BUILD_default}"
  CACHE STRING "Targets for BOLT to support.")
if (NOT BOLT_TARGETS_TO_BUILD)
  message(FATAL_ERROR "BOLT enabled but BOLT_TARGETS_TO_BUILD is empty")
endif()
foreach (tgt ${BOLT_TARGETS_TO_BUILD})
  if (NOT tgt IN_LIST LLVM_TARGETS_TO_BUILD)
    message(FATAL_ERROR "BOLT target '${tgt}' is not in LLVM_TARGETS_TO_BUILD")
  endif()
  message(STATUS "Targeting ${tgt} in llvm-bolt")
endforeach()

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

find_program(GNU_LD_EXECUTABLE NAMES ${LLVM_DEFAULT_TARGET_TRIPLE}-ld.bfd ld.bfd DOC "GNU ld")

include(AddBOLT)

option(BOLT_BUILD_TOOLS
  "Build the BOLT tools. If OFF, just generate build targets." ON)

add_custom_target(bolt)
set_target_properties(bolt PROPERTIES FOLDER "BOLT/Metatargets")
add_llvm_install_targets(install-bolt DEPENDS bolt COMPONENT bolt)

if (bolt-rt IN_LIST LLVM_ENABLE_RUNTIMES)
  add_dependencies(bolt bolt-rt)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

add_subdirectory(lib)
add_subdirectory(tools)

if (BOLT_INCLUDE_TESTS)
  if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/bolt/RuntimeLibs/RuntimeLibraryVariables.inc.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/bolt/RuntimeLibs/RuntimeLibraryVariables.inc @ONLY)

set(BOLT_ENUM_TARGETS "")
foreach(t ${BOLT_TARGETS_TO_BUILD})
  set(BOLT_ENUM_TARGETS "${BOLT_ENUM_TARGETS}BOLT_TARGET(${t})\n")
endforeach(t)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/bolt/Core/TargetConfig.def.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/bolt/Core/TargetConfig.def @ONLY)
