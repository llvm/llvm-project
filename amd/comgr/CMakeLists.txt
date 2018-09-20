cmake_minimum_required(VERSION 3.2.0)
project(code_object_manager)

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-rtti -Wall -Wno-attributes")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fms-extensions")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  # TODO: Confirm this is actually needed due to LLVM/Clang code
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
  endif()
  add_definitions(-D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)
if (APPLE)
  set(LINKER_OPTS "-pthread ")
else()
  set(LINKER_OPTS "-pthread -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap -Wl,--no-undefined")
endif()

else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
endif()

# Windows is strict about visibility of exports in shared libraries, so we ask
# GCC/Clang to also be strict, and then explicitly mark each exported symbol in
# the shared header.
add_definitions(-DAMD_EXPORT)

file(GLOB SOURCES "src/*.cpp")
add_library(comgr SHARED ${SOURCES})

install(
  TARGETS
  comgr
  DESTINATION lib
)

install(
  FILES
  "include/comgr/amd_comgr.h"
  DESTINATION include/comgr
)

install(
  FILES
  "README.md"
  "LICENSE.txt"
  "NOTICES.txt"
  DESTINATION comgr
)

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Enable testing" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Enable parse tools" FORCE)
add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)

include_directories(include)
include_directories(./yaml-cpp/include)

set(CLANG_LIBS
  clangFrontendTool
  clangARCMigrate
  clangRewriteFrontend
  clangStaticAnalyzerFrontend
  clangRewrite
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  clangCrossTU
  clangASTMatchers
  clangIndex
  clangDriver
  clangCodeGen
  clangFrontend
  clangSerialization
  clangParse
  clangSema
  clangEdit
  clangLex
  clangAnalysis
  clangAST
  clangBasic
)

set(LLD_LIBS
  lldELF
  lldCore
  lldCommon
)

llvm_map_components_to_libnames(LLVM_LIBS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  SelectionDAG
  GlobalISel
  CodeGen
  ipo
  Instrumentation
  ScalarOpts
  Vectorize
  InstCombine
  IRReader
  AsmParser
  Target
  Analysis
  Symbolize
  DebugInfoDWARF
  DebugInfoPDB
  DebugInfoCodeView
  DebugInfoMSF
  MCDisassembler
  Object
  MCParser
  MC
  TransformUtils
  BitReader
  Core
  BinaryFormat
  Support
  Demangle
  Option
  ProfileData
  Coverage
  LTO
  Coroutines
)

target_link_libraries(comgr ${LINKER_OPTS}
  yaml-cpp
  ${CLANG_LIBS}
  ${LLD_LIBS}
  ${LLVM_LIBS}
)

if (NOT UNIX)
  target_link_libraries(comgr version)
endif()

enable_testing()
add_subdirectory(test)

# Add packaging directives for comgr
set(CPACK_PACKAGE_NAME comgr)
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACKAGE_VERSION_MINOR 0)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Library to provide support functions")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE
  "https://github.com/ROCmSoftwarePlatform/ROCm-CodeObjectManager")

# RPM package specific variables
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

if (NOT CPack_CMake_INCLUDED)
  include(CPack)
endif()
