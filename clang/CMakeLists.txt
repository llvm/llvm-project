cmake_minimum_required(VERSION 3.4.3)

# If we are not building as a part of LLVM, build Clang as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(Clang)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
  else()
    message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
  endif()

  list(GET CONFIG_OUTPUT 0 ENABLE_ASSERTIONS)
  list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 INCLUDE_DIR)
  list(GET CONFIG_OUTPUT 4 LLVM_OBJ_ROOT)
  list(GET CONFIG_OUTPUT 5 MAIN_SRC_DIR)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()

  # They are used as destination of target generators.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  if(WIN32 OR CYGWIN)
    # DLL platform -- put DLLs into bin.
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  endif()

  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)
  option(CLANG_ENABLE_BOOTSTRAP "Generate the clang bootstrap target" OFF)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
  include(VersionFromVCS)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${LLVM_LIBRARY_DIR}")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX} )

  if(LLVM_INCLUDE_TESTS)
    set(Python_ADDITIONAL_VERSIONS 2.7)
    include(FindPythonInterp)
    if(NOT PYTHONINTERP_FOUND)
      message(FATAL_ERROR
"Unable to find Python interpreter, required for builds and testing.

Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
    endif()

    if( ${PYTHON_VERSION_STRING} VERSION_LESS 2.7 )
      message(FATAL_ERROR "Python 2.7 or newer is required")
    endif()

    # Check prebuilt llvm/utils.
    if(EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()

    if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      if(NOT LLVM_UTILS_PROVIDED)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/count utils/count)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/not utils/not)
        set(LLVM_UTILS_PROVIDED ON)
        set(CLANG_TEST_DEPS FileCheck count not)
      endif()
      set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
      if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
          AND NOT EXISTS ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
          AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
        add_subdirectory(${UNITTEST_DIR} utils/unittest)
      endif()
    else()
      # Seek installed Lit.
      find_program(LLVM_LIT "lit.py" ${LLVM_MAIN_SRC_DIR}/utils/lit
        DOC "Path to lit.py")
    endif()

    if(LLVM_LIT)
      # Define the default arguments to use with 'lit', and an option for the user
      # to override.
      set(LIT_ARGS_DEFAULT "-sv")
      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if( WIN32 AND NOT CYGWIN )
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()
    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()
  endif()

  set( CLANG_BUILT_STANDALONE 1 )
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

# Make sure that our source directory is on the current cmake module path so that
# we can include cmake files from this directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(LibXml2 2.5.3 QUIET)
if (LIBXML2_FOUND)
  set(CLANG_HAVE_LIBXML 1)
endif()

include(CheckIncludeFile)
check_include_file(sys/resource.h CLANG_HAVE_RLIMITS)

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed." )
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>." )

set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")

set(ENABLE_X86_RELAX_RELOCATIONS OFF CACHE BOOL
    "enable x86 relax relocations by default")

set(CLANG_DEFAULT_CXX_STDLIB "" CACHE STRING
  "Default C++ stdlib to use (\"libstdc++\" or \"libc++\", empty for platform default")
if (NOT(CLANG_DEFAULT_CXX_STDLIB STREQUAL "" OR
        CLANG_DEFAULT_CXX_STDLIB STREQUAL "libstdc++" OR
        CLANG_DEFAULT_CXX_STDLIB STREQUAL "libc++"))
  message(WARNING "Resetting default C++ stdlib to use platform default")
  set(CLANG_DEFAULT_CXX_STDLIB "" CACHE STRING
    "Default C++ stdlib to use (\"libstdc++\" or \"libc++\", empty for platform default" FORCE)
endif()

set(CLANG_DEFAULT_RTLIB "" CACHE STRING
  "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)")
if (NOT(CLANG_DEFAULT_RTLIB STREQUAL "" OR
        CLANG_DEFAULT_RTLIB STREQUAL "libgcc" OR
        CLANG_DEFAULT_RTLIB STREQUAL "compiler-rt"))
  message(WARNING "Resetting default rtlib to use platform default")
  set(CLANG_DEFAULT_RTLIB "" CACHE STRING
    "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)" FORCE)
endif()

set(CLANG_DEFAULT_OPENMP_RUNTIME "libomp" CACHE STRING
  "Default OpenMP runtime used by -fopenmp.")

set(CLANG_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific text for showing with version information.")

if( CLANG_VENDOR )
  add_definitions( -DCLANG_VENDOR="${CLANG_VENDOR} " )
endif()

set(CLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")

if(CLANG_REPOSITORY_STRING)
  add_definitions(-DCLANG_REPOSITORY_STRING="${CLANG_REPOSITORY_STRING}")
endif()

set(CLANG_VENDOR_UTI "org.llvm.clang" CACHE STRING
  "Vendor-specific uti.")

# The libdir suffix must exactly match whatever LLVM's configuration used.
set(CLANG_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")

set(CLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLVM. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${CLANG_SOURCE_DIR}/include/clang/*.inc")
  if( tablegenned_files_on_include_dir )
    message(FATAL_ERROR "Apparently there is a previous in-source build, "
"probably as the result of running `configure' and `make' on "
"${CLANG_SOURCE_DIR}. This may cause problems. The suspicious files are:\n"
"${tablegenned_files_on_include_dir}\nPlease clean the source directory.")
  endif()
endif()

# If CLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED CLANG_VERSION_MAJOR)
  set(CLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED CLANG_VERSION_MINOR)
  set(CLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED CLANG_VERSION_PATCHLEVEL)
  set(CLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
# Unlike PACKAGE_VERSION, CLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(CLANG_VERSION "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}.${CLANG_VERSION_PATCHLEVEL}")
message(STATUS "Clang version: ${CLANG_VERSION}")

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/clang/Basic/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/clang/Basic/Version.inc)

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual")
  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
  endif ()

  # Enable -pedantic for Clang even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wno-long-long")
  endif ()

  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if( CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types" )
  endif()
endif ()

# Determine HOST_LINK_VERSION on Darwin.
set(HOST_LINK_VERSION)
if (APPLE)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT
  )
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

configure_file(
  ${CLANG_SOURCE_DIR}/include/clang/Config/config.h.cmake
  ${CLANG_BINARY_DIR}/include/clang/Config/config.h)

include(CMakeParseArguments)
include(AddClang)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/clang include/clang-c
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".svn" EXCLUDE
    )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/clang
    DESTINATION include
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    PATTERN "*.h"
    )
endif()

add_definitions( -D_GNU_SOURCE )

option(CLANG_BUILD_TOOLS
  "Build the Clang tools. If OFF, just generate build targets." ON)

option(CLANG_ENABLE_ARCMT "Build ARCMT." ON)
if (CLANG_ENABLE_ARCMT)
  set(ENABLE_CLANG_ARCMT "1")
else()
  set(ENABLE_CLANG_ARCMT "0")
endif()

option(CLANG_ENABLE_STATIC_ANALYZER "Build static analyzer." ON)
if (CLANG_ENABLE_STATIC_ANALYZER)
  set(ENABLE_CLANG_STATIC_ANALYZER "1")
else()
  set(ENABLE_CLANG_STATIC_ANALYZER "0")
endif()

if (NOT CLANG_ENABLE_STATIC_ANALYZER AND CLANG_ENABLE_ARCMT)
  message(FATAL_ERROR "Cannot disable static analyzer while enabling ARCMT")
endif()

if(CLANG_ENABLE_ARCMT)
  add_definitions(-DCLANG_ENABLE_ARCMT)
  add_definitions(-DCLANG_ENABLE_OBJC_REWRITER)
endif()
if(CLANG_ENABLE_STATIC_ANALYZER)
  add_definitions(-DCLANG_ENABLE_STATIC_ANALYZER)
endif()

# Clang version information
set(CLANG_EXECUTABLE_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the clang executable, in the form XX.YY")
set(LIBCLANG_LIBRARY_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the libclang library , in the form XX.YY")
mark_as_advanced(CLANG_EXECUTABLE_VERSION LIBCLANG_LIBRARY_VERSION)

option(CLANG_INCLUDE_TESTS
       "Generate build targets for the Clang unit tests."
       ${LLVM_INCLUDE_TESTS})

add_subdirectory(utils/TableGen)

add_subdirectory(include)

# All targets below may depend on all tablegen'd files.
get_property(CLANG_TABLEGEN_TARGETS GLOBAL PROPERTY CLANG_TABLEGEN_TARGETS)
list(APPEND LLVM_COMMON_DEPENDS ${CLANG_TABLEGEN_TARGETS})

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

option(CLANG_BUILD_EXAMPLES "Build CLANG example programs by default." OFF)
if (CLANG_BUILD_EXAMPLES)
  set(ENABLE_CLANG_EXAMPLES "1")
else()
  set(ENABLE_CLANG_EXAMPLES "0")
endif()
add_subdirectory(examples)

if( CLANG_INCLUDE_TESTS )
  if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(unittests)
    list(APPEND CLANG_TEST_DEPS ClangUnitTests)
    list(APPEND CLANG_TEST_PARAMS
      clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
      )
  endif()
  add_subdirectory(test)

  if(CLANG_BUILT_STANDALONE)
    # Add a global check rule now that all subdirectories have been traversed
    # and we know the total set of lit testsuites.
    get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
    get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
    get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
    get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
    add_lit_target(check-all
      "Running all regression tests"
      ${LLVM_LIT_TESTSUITES}
      PARAMS ${LLVM_LIT_PARAMS}
      DEPENDS ${LLVM_LIT_DEPENDS}
      ARGS ${LLVM_LIT_EXTRA_ARGS}
      )
  endif()
  add_subdirectory(utils/perf-training)
endif()

option(CLANG_INCLUDE_DOCS "Generate build targets for the Clang docs."
  ${LLVM_INCLUDE_DOCS})
if( CLANG_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()


if(APPLE)
  # this line is needed as a cleanup to ensure that any CMakeCaches with the old
  # default value get updated to the new default.
  if(CLANG_ORDER_FILE STREQUAL "")
    unset(CLANG_ORDER_FILE CACHE)
    unset(CLANG_ORDER_FILE)
  endif()


  set(CLANG_ORDER_FILE ${CMAKE_CURRENT_BINARY_DIR}/clang.order CACHE FILEPATH
    "Order file to use when compiling clang in order to improve startup time (Darwin Only - requires ld64).")

  if(CLANG_ORDER_FILE AND NOT EXISTS ${CLANG_ORDER_FILE})
    string(FIND "${CLANG_ORDER_FILE}" "${CMAKE_CURRENT_BINARY_DIR}" PATH_START)
    if(PATH_START EQUAL 0)
      file(WRITE ${CLANG_ORDER_FILE} "\n")
    else()
      message(FATAL_ERROR "Specified order file '${CLANG_ORDER_FILE}' does not exist.")
    endif()
  endif()
endif()

add_subdirectory(cmake/modules)

if(CLANG_STAGE)
  message(STATUS "Setting current clang stage to: ${CLANG_STAGE}")
endif()

if (CLANG_ENABLE_BOOTSTRAP)
  include(ExternalProject)

  if(NOT CLANG_STAGE)
    set(CLANG_STAGE stage1)
  endif()

  string(REGEX MATCH "stage([0-9]*)" MATCHED_STAGE "${CLANG_STAGE}")
  if(MATCHED_STAGE)
    if(NOT LLVM_BUILD_INSTRUMENTED)
      math(EXPR STAGE_NUM "${CMAKE_MATCH_1} + 1")
      set(NEXT_CLANG_STAGE stage${STAGE_NUM})
    else()
      set(NEXT_CLANG_STAGE stage${CMAKE_MATCH_1})
    endif()
  else()
    set(NEXT_CLANG_STAGE bootstrap)
  endif()

  if(BOOTSTRAP_LLVM_BUILD_INSTRUMENTED)
    set(NEXT_CLANG_STAGE ${NEXT_CLANG_STAGE}-instrumented)
  endif()
  message(STATUS "Setting next clang stage to: ${NEXT_CLANG_STAGE}")
  
  
  set(STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NEXT_CLANG_STAGE}-stamps/)
  set(BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NEXT_CLANG_STAGE}-bins/)

  # If the next stage is LTO we need to depend on LTO and possibly LLVMgold
  if(BOOTSTRAP_LLVM_ENABLE_LTO OR LLVM_ENABLE_LTO)
    set(LTO_DEP LTO)
    if(APPLE)
      # on Darwin we need to set DARWIN_LTO_LIBRARY so that -flto will work
      # using the just-built compiler, and we need to override DYLD_LIBRARY_PATH
      # so that the host object file tools will use the just-built libLTO.
      # However if System Integrity Protection is enabled the DYLD variables
      # will be scrubbed from the environment of any base system commands. This
      # includes /bin/sh, which ninja uses when executing build commands. To
      # work around the envar being filtered away we pass it in as a CMake
      # variable, and have LLVM's CMake append the envar to the archiver calls.
      set(LTO_LIBRARY -DDARWIN_LTO_LIBRARY=${LLVM_SHLIB_OUTPUT_INTDIR}/libLTO.dylib
        -DDYLD_LIBRARY_PATH=${LLVM_LIBRARY_OUTPUT_INTDIR})
    elseif(NOT WIN32)
      list(APPEND LTO_DEP LLVMgold llvm-ar llvm-ranlib)
      set(LTO_AR -DCMAKE_AR=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ar)
      set(LTO_RANLIB -DCMAKE_RANLIB=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ranlib)
    endif()
  endif()

  add_custom_target(${NEXT_CLANG_STAGE}-clear
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${NEXT_CLANG_STAGE}-cleared
    )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NEXT_CLANG_STAGE}-cleared
    DEPENDS clang ${LTO_DEP}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${STAMP_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${STAMP_DIR}
    COMMENT "Clobberring ${NEXT_CLANG_STAGE} build and stamp directories"
    )

  if(CMAKE_VERBOSE_MAKEFILE)
    set(verbose -DCMAKE_VERBOSE_MAKEFILE=On)
  endif()

  set(_BOOTSTRAP_DEFAULT_PASSTHROUGH
    PACKAGE_VERSION
    LLVM_VERSION_MAJOR
    LLVM_VERSION_MINOR
    LLVM_VERSION_PATCH
    CLANG_VERSION_MAJOR
    CLANG_VERSION_MINOR
    CLANG_VERSION_PATCHLEVEL
    LLVM_VERSION_SUFFIX
    LLVM_BINUTILS_INCDIR
    CLANG_REPOSITORY_STRING
    CMAKE_MAKE_PROGRAM)

  if(TARGET compiler-rt)
    set(RUNTIME_DEP compiler-rt)
  endif()

  set(COMPILER_OPTIONS
    -DCMAKE_CXX_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang++
    -DCMAKE_C_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang
    -DCMAKE_ASM_COMPILER=${LLVM_RUNTIME_OUTPUT_INTDIR}/clang)

  if(BOOTSTRAP_LLVM_BUILD_INSTRUMENTED)
    set(PGO_DEP llvm-profdata)
    set(PGO_OPT -DLLVM_PROFDATA=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-profdata)
  endif()

  if(LLVM_BUILD_INSTRUMENTED)
    set(PGO_DEP generate-profdata)
    set(PGO_OPT -DLLVM_PROFDATA_FILE=${CMAKE_CURRENT_BINARY_DIR}/utils/perf-training/clang.profdata)
    # Use the current tools for LTO instead of the instrumented ones
    list(APPEND _BOOTSTRAP_DEFAULT_PASSTHROUGH
      CMAKE_CXX_COMPILER
      CMAKE_C_COMPILER
      CMAKE_ASM_COMPILER
      CMAKE_AR
      CMAKE_RANLIB
      DARWIN_LTO_LIBRARY
      DYLD_LIBRARY_PATH)

    set(COMPILER_OPTIONS)
    set(LTO_LIBRARY)
    set(RUNTIME_DEP) # Don't set runtime dependencies
    set(LTO_DEP)     # Don't need to depend on LTO
    set(LTO_AR)
    set(LTO_RANLIB)
  endif()

  # Find all variables that start with BOOTSTRAP_ and populate a variable with
  # them.
  get_cmake_property(variableNames VARIABLES)
  foreach(variableName ${variableNames})
    if(variableName MATCHES "^BOOTSTRAP_")
      string(SUBSTRING ${variableName} 10 -1 varName)
      string(REPLACE ";" "\;" value "${${variableName}}")
      list(APPEND PASSTHROUGH_VARIABLES
        -D${varName}=${value})
    endif()
    if(${variableName} AND variableName MATCHES "LLVM_EXTERNAL_.*_SOURCE_DIR")
      list(APPEND PASSTHROUGH_VARIABLES
        -D${variableName}=${${variableName}})
    endif()
  endforeach()

  # Populate the passthrough variables
  foreach(variableName ${CLANG_BOOTSTRAP_PASSTHROUGH} ${_BOOTSTRAP_DEFAULT_PASSTHROUGH})
    if(${variableName})
      string(REPLACE ";" "\;" value ${${variableName}})
      list(APPEND PASSTHROUGH_VARIABLES
        -D${variableName}=${value})
    endif()
  endforeach()

  ExternalProject_Add(${NEXT_CLANG_STAGE}
    DEPENDS clang ${LTO_DEP} ${RUNTIME_DEP} ${PGO_DEP}
    PREFIX ${NEXT_CLANG_STAGE}
    SOURCE_DIR ${CMAKE_SOURCE_DIR}
    STAMP_DIR ${STAMP_DIR}
    BINARY_DIR ${BINARY_DIR}
    EXCLUDE_FROM_ALL 1
    CMAKE_ARGS
                # We shouldn't need to set this here, but INSTALL_DIR doesn't
                # seem to work, so instead I'm passing this through
                -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                ${CLANG_BOOTSTRAP_CMAKE_ARGS}
                ${PASSTHROUGH_VARIABLES}
                 -DCLANG_STAGE=${NEXT_CLANG_STAGE}
                ${COMPILER_OPTIONS}
                ${LTO_LIBRARY} ${LTO_AR} ${LTO_RANLIB} ${verbose} ${PGO_OPT}
    INSTALL_COMMAND ""
    STEP_TARGETS configure build
    USES_TERMINAL_CONFIGURE 1
    USES_TERMINAL_BUILD 1
    USES_TERMINAL_INSTALL 1
    )

  # exclude really-install from main target
  set_target_properties(${NEXT_CLANG_STAGE} PROPERTIES _EP_really-install_EXCLUDE_FROM_MAIN On)
  ExternalProject_Add_Step(${NEXT_CLANG_STAGE} really-install
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    COMMENT "Performing install step for '${NEXT_CLANG_STAGE}'"
    DEPENDEES build
    USES_TERMINAL 1
  )
  ExternalProject_Add_StepTargets(${NEXT_CLANG_STAGE} really-install)
  add_custom_target(${NEXT_CLANG_STAGE}-install DEPENDS ${NEXT_CLANG_STAGE}-really-install)

  if(NOT CLANG_BOOTSTRAP_TARGETS)
    set(CLANG_BOOTSTRAP_TARGETS check-llvm check-clang check-all)
  endif()
  foreach(target ${CLANG_BOOTSTRAP_TARGETS})
    # exclude from main target
    set_target_properties(${NEXT_CLANG_STAGE} PROPERTIES _EP_${target}_EXCLUDE_FROM_MAIN On)

    ExternalProject_Add_Step(${NEXT_CLANG_STAGE} ${target}
      COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target ${target}
      COMMENT "Performing ${target} for '${NEXT_CLANG_STAGE}'"
      DEPENDEES configure
      USES_TERMINAL 1
    )

    if(target MATCHES "^stage[0-9]*")
      add_custom_target(${target} DEPENDS ${NEXT_CLANG_STAGE}-${target})
    endif()

    ExternalProject_Add_StepTargets(${NEXT_CLANG_STAGE} ${target})
  endforeach()
endif()

if (LLVM_ADD_NATIVE_VISUALIZERS_TO_SOLUTION)
  add_subdirectory(utils/ClangVisualizers)
endif()
