cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# Flang requires C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. \
          Please create a directory and run cmake from there,\
          passing the path to this source directory as the last argument.\
          This process created the file `CMakeCache.txt' and the directory\
          `CMakeFiles'. Please delete them.")
endif()

option(FLANG_ENABLE_WERROR "Fail and stop building flang if a warning is triggered." OFF)

# Check for a standalone build and configure as appropriate from
# there.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building Flang as a standalone project.")
  project(Flang)
  set(FLANG_STANDALONE_BUILD ON)
else()
  set(FLANG_STANDALONE_BUILD OFF)
endif()

# Must go below project(..)
include(GNUInstallDirs)

if (FLANG_STANDALONE_BUILD)
  set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  if (NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions follow llvm's configuration.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  # We need a pre-built/installed version of LLVM.
  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  # If the user specifies a relative path to LLVM_DIR, the calls to include
  # LLVM modules fail. Append the absolute path to LLVM_DIR instead.
  get_filename_component(LLVM_DIR_ABSOLUTE ${LLVM_DIR} REALPATH)
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR_ABSOLUTE})

  # Users might specify a path to CLANG_DIR that's:
  #   * a full path, or
  #   * a path relative to the path of this script.
  # Append the absolute path to CLANG_DIR so that find_package works in both
  # cases.
  get_filename_component(
    CLANG_DIR_ABSOLUTE
    ${CLANG_DIR}
    REALPATH
    ${CMAKE_CURRENT_SOURCE_DIR})
  list(APPEND CMAKE_MODULE_PATH ${CLANG_DIR_ABSOLUTE})

  # TODO: Remove when libclangDriver is lifted out of Clang
  find_package(Clang REQUIRED PATHS "${CLANG_DIR_ABSOLUTE}" NO_DEFAULT_PATH)
  if (NOT Clang_FOUND)
    message(FATAL_ERROR "Failed to find Clang")
  endif()

  # If LLVM links to zlib we need the imported targets so we can too.
  if(LLVM_ENABLE_ZLIB)
    find_package(ZLIB REQUIRED)
  endif()
  option(LLVM_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(USE_NO_MAYBE_UNINITIALIZED 1)
  endif()

  include(CMakeParseArguments)
  include(AddLLVM)
  include(HandleLLVMOptions)
  include(VersionFromVCS)
  include(GetErrcMessages)

  include(AddClang)

  include(TableGen)
  find_package(MLIR REQUIRED CONFIG)
  # Use SYSTEM for the same reasons as for LLVM includes
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
  # If the user specifies a relative path to MLIR_DIR, the calls to include
  # MLIR modules fail. Append the absolute path to MLIR_DIR instead.
  get_filename_component(MLIR_DIR_ABSOLUTE ${MLIR_DIR} REALPATH)
  list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR_ABSOLUTE})
  include(AddMLIR)
  find_program(MLIR_TABLEGEN_EXE "mlir-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)
  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)


  # Add LLVM include files as if they were SYSTEM because there are complex unused
  # parameter issues that may or may not appear depending on the environments and
  # compilers (ifdefs are involved). This allows warnings from LLVM headers to be
  # ignored while keeping -Wunused-parameter a fatal error inside f18 code base.
  # This may have to be fine-tuned if flang headers are consider part of this
  # LLVM_INCLUDE_DIRS when merging in the monorepo (Warning from flang headers
  # should not be suppressed).
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  # LLVM's cmake configuration files currently sneak in a c++11 flag.
  # We look for it here and remove it from Flang's compile flags to
  # avoid some mixed compilation flangs (e.g. -std=c++11 ... -std=c++17).
  if (DEFINED LLVM_CXX_STD)
    message("LLVM configuration set a C++ standard: ${LLVM_CXX_STD}")
    if (NOT LLVM_CXX_STD EQUAL "c++17")
      message("Flang: Overriding LLVM's 'cxx_std' setting...")
      message("    removing '-std=${LLVM_CXX_STD}'")
      message("    CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}'")
      string(REPLACE " -std=${LLVM_CXX_STD}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      message("    [NEW] CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}'")
    endif()
  endif()

  link_directories("${LLVM_LIBRARY_DIR}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit" CACHE STRING "Command used to spawn lit")

  option(FLANG_INCLUDE_TESTS
         "Generate build targets for the Flang unit tests."
         ON)

  get_errc_messages(LLVM_LIT_ERRC_MESSAGES)

#Handle unittests when out-of-tree
#LLVM_BUILD_MAIN_SRC_DIR - Path to llvm source when out-of-tree.
  set(FLANG_GTEST_AVAIL 0)
  if (FLANG_INCLUDE_TESTS)
    set(UNITTEST_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/utils/unittest)
    if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
      if (NOT TARGET llvm_gtest)
        add_library(llvm_gtest
          ${UNITTEST_DIR}/googletest/src/gtest-all.cc
          ${UNITTEST_DIR}/googlemock/src/gmock-all.cc
          )
        target_include_directories(llvm_gtest
          PUBLIC
          "${UNITTEST_DIR}/googletest/include"
          "${UNITTEST_DIR}/googlemock/include"

          PRIVATE
          "${UNITTEST_DIR}/googletest"
          "${UNITTEST_DIR}/googlemock"
          )
         find_package(Threads)
         target_link_libraries(llvm_gtest PUBLIC Threads::Threads)
        add_library(llvm_gtest_main ${UNITTEST_DIR}/UnitTestMain/TestMain.cpp)
        target_link_libraries(gtest_main PUBLIC llvm_gtest)
      endif()
      set(FLANG_GTEST_AVAIL 1)
    else()
      message(WARNING
      "Unit-tests will be skipped as LLVM install does not include google-test related headers and libraries.")
      set(FLANG_GTEST_AVAIL 0)
    endif()
  endif()
  if (FLANG_GTEST_AVAIL)
    add_custom_target(check-all DEPENDS check-flang FlangUnitTests)
  else()
    add_custom_target(check-all DEPENDS check-flang )
  endif()
  if (LLVM_BUILD_DOCS)
    add_custom_target(doxygen ALL)
  endif()

else()
  option(FLANG_INCLUDE_TESTS
         "Generate build targets for the Flang unit tests."
         ${LLVM_INCLUDE_TESTS})
  set(FLANG_GTEST_AVAIL 1)

  if(FLANG_STANDALONE_BUILD)
    set(FLANG_BINARY_DIR ${CMAKE_BINARY_DIR}/tools/flang)
  else()
    set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
endif()

set(FLANG_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(FLANG_TOOLS_INSTALL_DIR)

set(FLANG_INTRINSIC_MODULES_DIR ${CMAKE_BINARY_DIR}/include/flang)
set(FLANG_INCLUDE_DIR ${FLANG_BINARY_DIR}/include)

# TODO: Remove when libclangDriver is lifted out of Clang
if(FLANG_STANDALONE_BUILD)
  set(CLANG_INCLUDE_DIR ${CLANG_INCLUDE_DIRS} )
  # No need to specify TableGen output dir as that's embedded in CLANG_DIR
else()
  set(CLANG_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../clang/include )
  # Specify TableGen output dir for things like DiagnosticCommonKinds.inc,
  # DiagnosticDriverKinds.inc (required for reporting diagnostics)
  set(CLANG_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/clang/include)
  include_directories(SYSTEM ${CLANG_TABLEGEN_OUTPUT_DIR})
endif()
include_directories(SYSTEM ${CLANG_INCLUDE_DIR})

# tco tool and FIR lib output directories
if(FLANG_STANDALONE_BUILD)
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
endif()
# Always build tco tool
set(LLVM_BUILD_TOOLS ON)

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include)

if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${FLANG_SOURCE_DIR}/../cmake)
endif()

# Add Flang-centric modules to cmake path.
list(INSERT CMAKE_MODULE_PATH 0
  "${FLANG_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )
include(AddFlang)

if (NOT DEFAULT_SYSROOT)
  set(DEFAULT_SYSROOT "" CACHE PATH
    "The <path> to use for the system root for all compiler invocations (--sysroot=<path>).")
endif()

if (NOT ENABLE_LINKER_BUILD_ID)
  set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")
endif()

set(FLANG_DEFAULT_LINKER "" CACHE STRING
  "Default linker to use (linker name or absolute path, empty for platform default)")

set(FLANG_DEFAULT_RTLIB "" CACHE STRING
   "Default Fortran runtime library to use (\"libFortranRuntime\"), leave empty for platform default.")

if (NOT(FLANG_DEFAULT_RTLIB STREQUAL ""))
  message(WARNING "Resetting Flang's default runtime library to use platform default.")
  set(FLANG_DEFAULT_RTLIB "" CACHE STRING
      "Default runtime library to use (empty for platform default)" FORCE)
endif()



set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")


if (NOT DEFINED FLANG_VERSION_MAJOR)
  set(FLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()

if (NOT DEFINED FLANG_VERSION_MINOR)
  set(FLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()

if (NOT DEFINED FLANG_VERSION_PATCHLEVEL)
  set(FLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()

# Unlike PACKAGE_VERSION, FLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "Flang version: ${FLANG_VERSION}")
# Flang executable version information
set(FLANG_EXECUTABLE_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number to appended to the flang executable name.")
set(LIBFLANG_LIBRARY_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number to appended to the libflang library.")

mark_as_advanced(FLANG_EXECUTABLE_VERSION LIBFLANG_LIBRARY_VERSION)

set(FLANG_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific Flang version information.")
set(FLANG_VENDOR_UTI "org.llvm.flang" CACHE STRING
  "Vendor-specific uti.")

if (FLANG_VENDOR)
  add_definitions(-DFLANG_VENDOR="${FLANG_VENDOR} ")
endif()

set(FLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")
if (FLANG_REPOSITORY_STRING)
  add_definitions(-DFLANG_REPOSITORY_STRING="${FLANG_REPOSITORY_STRING}")
endif()

include(TestBigEndian)
test_big_endian(IS_BIGENDIAN)
if (IS_BIGENDIAN)
  add_compile_definitions(FLANG_BIG_ENDIAN=1)
else ()
  add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
endif ()

# Configure Flang's Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/flang/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/flang/Version.inc)
# Configure Flang's version info header file.
configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)

if (FLANG_ENABLE_WERROR)
  # The following is taken from llvm/cmake/modules/HandleLLVMOptions.cmake
  # Keep this up-to-date with that file
  if( MSVC )
    append("/WX" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
  if ( LLVM_COMPILER_IS_GCC_COMPATIBLE )
    append("-Werror" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    append("-Wno-error" CMAKE_REQUIRED_FLAGS)
  endif( LLVM_COMPILER_IS_GCC_COMPATIBLE )
endif()

# Builtin check_cxx_compiler_flag doesn't seem to work correctly
macro(check_compiler_flag flag resultVar)
  unset(${resultVar} CACHE)
  check_cxx_compiler_flag("${flag}" ${resultVar})
endmacro()

check_compiler_flag("-Werror -Wno-deprecated-copy" CXX_SUPPORTS_NO_DEPRECATED_COPY_FLAG)
if (CXX_SUPPORTS_NO_DEPRECATED_COPY_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
endif()
check_compiler_flag("-Wstring-conversion" CXX_SUPPORTS_NO_STRING_CONVERSION_FLAG)
if (CXX_SUPPORTS_NO_STRING_CONVERSION_FLAG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion")
endif()

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)

  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -fno-semantic-interposition")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wstring-conversion \
          -Wcovered-switch-default")
  endif()  # Clang.

  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types")
  endif()

  # Add to build type flags.
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUGF18")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DCHECK=\"(void)\"")

  # Building shared libraries is bad for performance with GCC by default
  # due to the need to preserve the right to override external entry points
  if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
  endif()

endif()

list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)

# Determine HOST_LINK_VERSION on Darwin.
set(HOST_LINK_VERSION)
if (APPLE)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT)
  if (NOT HAD_ERROR)
    if ("${LD_V_OUTPUT}" MATCHES ".*ld64-([0-9.]+).*")
      string(REGEX REPLACE ".*ld64-([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    elseif ("${LD_V_OUTPUT}" MATCHES "[^0-9]*([0-9.]+).*")
      string(REGEX REPLACE "[^0-9]*([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    endif()
  else()
    message(FATAL_ERROR "${CMAKE_LINKER} failed with status ${HAD_ERROR}")
  endif()
endif()

include(CMakeParseArguments)
include(AddFlang)


add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(cmake/modules)

option(FLANG_BUILD_TOOLS
  "Build the Flang tools. If OFF, just generate build targets." ON)
if (FLANG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()
add_subdirectory(runtime)

option(FLANG_BUILD_EXAMPLES "Build Flang example programs by default." OFF)
if (FLANG_BUILD_EXAMPLES AND FLANG_STANDALONE_BUILD)
  message(FATAL_ERROR "Examples are not supported in out-of-tree builds.")
endif()
add_subdirectory(examples)

if (FLANG_INCLUDE_TESTS)
  add_subdirectory(test)
  if (FLANG_GTEST_AVAIL)
    add_subdirectory(unittests)
  endif ()
endif()

option(FLANG_INCLUDE_DOCS "Generate build targets for the Flang docs."
       ${LLVM_INCLUDE_DOCS})
if (FLANG_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

# Custom target to install Flang libraries.
add_custom_target(flang-libraries)
set_target_properties(flang-libraries PROPERTIES FOLDER "Misc")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-flang-libraries
   DEPENDS flang-libraries
   COMPONENT flang-libraries)
endif()

get_property(FLANG_LIBS GLOBAL PROPERTY FLANG_LIBS)
if (FLANG_LIBS)
  list(REMOVE_DUPLICATES FLANG_LIBS)
  foreach(lib ${FLANG_LIBS})
    add_dependencies(flang-libraries ${lib})
    if (NOT LLVM_ENABLE_IDE)
      add_dependencies(install-flang-libraries install-${lib})
    endif()
  endforeach()
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/flang
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "config.h" EXCLUDE
    PATTERN ".git"     EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE)

  install(DIRECTORY ${FLANG_INCLUDE_DIR}/flang
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.inc"
    )
endif()
