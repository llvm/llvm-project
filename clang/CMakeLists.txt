cmake_minimum_required(VERSION 2.8.8)

# FIXME: It may be removed when we use 2.8.12.
if(CMAKE_VERSION VERSION_LESS 2.8.12)
  # Invalidate a couple of keywords.
  set(cmake_2_8_12_INTERFACE)
  set(cmake_2_8_12_PRIVATE)
else()
  # Use ${cmake_2_8_12_KEYWORD} intead of KEYWORD in target_link_libraries().
  set(cmake_2_8_12_INTERFACE INTERFACE)
  set(cmake_2_8_12_PRIVATE PRIVATE)
  if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW) # automatic when 2.8.12 is required
  endif()
endif()

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

  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/share/llvm/cmake")
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

find_package(LibXml2 2.5.3 QUIET)
if (LIBXML2_FOUND)
  set(CLANG_HAVE_LIBXML 1)
endif()

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed." )
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>." )

set(CLANG_DEFAULT_OPENMP_RUNTIME "libomp" CACHE STRING
  "Default OpenMP runtime used by -fopenmp.")

set(CLANG_VENDOR "" CACHE STRING
  "Vendor-specific text for showing with version information.")

if( CLANG_VENDOR )
  add_definitions( -DCLANG_VENDOR="${CLANG_VENDOR} " )
endif()

set(CLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")

if(CLANG_REPOSITORY_STRING)
  add_definitions(-DCLANG_REPOSITORY_STRING="${CLANG_REPOSITORY_STRING}")
endif()

option(CLANG_APPEND_VC_REV
  "Append the version control system revision id to clang version spew" OFF)
if(CLANG_APPEND_VC_REV)
  if(NOT SVN_REVISION)
    # This macro will set SVN_REVISION in the parent scope
    add_version_info_from_vcs(VERSION_VAR)
  endif()

  if(SVN_REVISION)
    add_definitions(-DSVN_REVISION="${SVN_REVISION}")
  endif()
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

# Compute the Clang version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
  ${PACKAGE_VERSION})
message(STATUS "Clang version: ${CLANG_VERSION}")

string(REGEX REPLACE "([0-9]+)\\.[0-9]+(\\.[0-9]+)?" "\\1" CLANG_VERSION_MAJOR
  ${CLANG_VERSION})
string(REGEX REPLACE "[0-9]+\\.([0-9]+)(\\.[0-9]+)?" "\\1" CLANG_VERSION_MINOR
  ${CLANG_VERSION})
if (${CLANG_VERSION} MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
  set(CLANG_HAS_VERSION_PATCHLEVEL 1)
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" CLANG_VERSION_PATCHLEVEL
    ${CLANG_VERSION})
else()
  set(CLANG_HAS_VERSION_PATCHLEVEL 0)
endif()

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

function(clang_tablegen)
  # Syntax:
  # clang_tablegen output-file [tablegen-arg ...] SOURCE source-file
  # [[TARGET cmake-target-name] [DEPENDS extra-dependency ...]]
  #
  # Generates a custom command for invoking tblgen as
  #
  # tblgen source-file -o=output-file tablegen-arg ...
  #
  # and, if cmake-target-name is provided, creates a custom target for
  # executing the custom command depending on output-file. It is
  # possible to list more files to depend after DEPENDS.

  cmake_parse_arguments(CTG "" "SOURCE;TARGET" "" ${ARGN})

  if( NOT CTG_SOURCE )
    message(FATAL_ERROR "SOURCE source-file required by clang_tablegen")
  endif()

  set( LLVM_TARGET_DEFINITIONS ${CTG_SOURCE} )
  tablegen(CLANG ${CTG_UNPARSED_ARGUMENTS})

  if(CTG_TARGET)
    add_public_tablegen_target(${CTG_TARGET})
    set_target_properties( ${CTG_TARGET} PROPERTIES FOLDER "Clang tablegenning")
    set_property(GLOBAL APPEND PROPERTY CLANG_TABLEGEN_TARGETS ${CTG_TARGET})
  endif()
endfunction(clang_tablegen)

macro(set_clang_windows_version_resource_properties name)
  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file}
      VERSION_MAJOR ${CLANG_VERSION_MAJOR}
      VERSION_MINOR ${CLANG_VERSION_MINOR}
      VERSION_PATCHLEVEL ${CLANG_VERSION_PATCHLEVEL}
      VERSION_STRING "${CLANG_VERSION} (${BACKEND_PACKAGE_STRING})"
      PRODUCT_NAME "clang")
  endif()
endmacro()

macro(add_clang_subdirectory name)
  add_llvm_subdirectory(CLANG TOOL ${name})
endmacro()

macro(add_clang_library name)
  cmake_parse_arguments(ARG
    "SHARED"
    ""
    "ADDITIONAL_HEADERS"
    ${ARGN})
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path
      ${CLANG_SOURCE_DIR}/lib/
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file( GLOB_RECURSE headers
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.h
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file( GLOB_RECURSE tds
        ${CLANG_SOURCE_DIR}/include/clang/${lib_path}/*.td
      )
      source_group("TableGen descriptions" FILES ${tds})
      set_source_files_properties(${tds}} PROPERTIES HEADER_FILE_ONLY ON)

      if(headers OR tds)
        set(srcs ${headers} ${tds})
      endif()
    endif()
  endif(MSVC_IDE OR XCODE)
  if(srcs OR ARG_ADDITIONAL_HEADERS)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARG_ADDITIONAL_HEADERS} # It may contain unparsed unknown args.
      )
  endif()
  if(ARG_SHARED)
    set(ARG_ENABLE_SHARED SHARED)
  endif()
  llvm_add_library(${name} ${ARG_ENABLE_SHARED} ${ARG_UNPARSED_ARGUMENTS} ${srcs})

  if(TARGET ${name})
    target_link_libraries(${name} ${cmake_2_8_12_INTERFACE} ${LLVM_COMMON_LIBS})

    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "libclang")
      install(TARGETS ${name}
        COMPONENT ${name}
        EXPORT ClangTargets
        LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
        RUNTIME DESTINATION bin)

      if (${ARG_SHARED} AND NOT CMAKE_CONFIGURATION_TYPES)
        add_custom_target(install-${name}
                          DEPENDS ${name}
                          COMMAND "${CMAKE_COMMAND}"
                                  -DCMAKE_INSTALL_COMPONENT=${name}
                                  -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      endif()
    endif()
    set_property(GLOBAL APPEND PROPERTY CLANG_EXPORTS ${name})
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Clang libraries")
  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "Clang executables")
  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_executable)

macro(add_clang_symlink name dest)
  add_llvm_tool_symlink(${name} ${dest} ALWAYS_GENERATE)
  # Always generate install targets
  llvm_install_symlink(${name} ${dest} ALWAYS_GENERATE)
endmacro()

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

set(CLANG_ORDER_FILE "" CACHE FILEPATH
  "Order file to use when compiling clang in order to improve startup time.")

if (CLANG_BUILT_STANDALONE OR CMAKE_VERSION VERSION_EQUAL 3 OR
    CMAKE_VERSION VERSION_GREATER 3)
  # Generate a list of CMake library targets so that other CMake projects can
  # link against them. LLVM calls its version of this file LLVMExports.cmake, but
  # the usual CMake convention seems to be ${Project}Targets.cmake.
  set(CLANG_INSTALL_PACKAGE_DIR share/clang/cmake)
  set(clang_cmake_builddir "${CMAKE_BINARY_DIR}/${CLANG_INSTALL_PACKAGE_DIR}")
  get_property(CLANG_EXPORTS GLOBAL PROPERTY CLANG_EXPORTS)
  export(TARGETS ${CLANG_EXPORTS} FILE ${clang_cmake_builddir}/ClangTargets.cmake)

  # Install a <prefix>/share/clang/cmake/ClangConfig.cmake file so that
  # find_package(Clang) works. Install the target list with it.
  install(EXPORT ClangTargets DESTINATION ${CLANG_INSTALL_PACKAGE_DIR})

  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ClangConfig.cmake
    DESTINATION share/clang/cmake)

  # Also copy ClangConfig.cmake to the build directory so that dependent projects
  # can build against a build directory of Clang more easily.
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ClangConfig.cmake
    ${CLANG_BINARY_DIR}/share/clang/cmake/ClangConfig.cmake
    COPYONLY)
endif ()

if (CLANG_ENABLE_BOOTSTRAP)
  include(ExternalProject)

  if(CMAKE_VERSION VERSION_GREATER 3.1.0)
    set(cmake_3_1_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL 1)
  endif()

  if(CMAKE_VERSION VERSION_GREATER 3.3.20150708)
    set(cmake_3_4_USES_TERMINAL_OPTIONS
      USES_TERMINAL_CONFIGURE 1
      USES_TERMINAL_BUILD 1
      USES_TERMINAL_INSTALL 1
      )
    set(cmake_3_4_USES_TERMINAL USES_TERMINAL 1)
  endif()

  if(NOT CLANG_STAGE)
    set(CLANG_STAGE stage1)
    message(STATUS "Setting current clang stage to: ${CLANG_STAGE}")
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

  # If on Darwin we need to make bootstrap depend on LTO and pass
  # DARWIN_LTO_LIBRARY so that -flto will work using the just-built compiler
  if(APPLE)
    set(LTO_DEP LTO llvm-ar llvm-ranlib)
    set(LTO_LIBRARY -DDARWIN_LTO_LIBRARY=${LLVM_SHLIB_OUTPUT_INTDIR}/libLTO.dylib)
    set(LTO_AR -DCMAKE_AR=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ar)
    set(LTO_RANLIB -DCMAKE_RANLIB=${LLVM_RUNTIME_OUTPUT_INTDIR}/llvm-ranlib)
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

  set(BOOTSTRAP_DEFAULT_PASSTHROUGH
    PACKAGE_VERSION
    LLVM_VERSION_MAJOR
    LLVM_VERSION_MINOR
    LLVM_VERSION_PATCH
    LLVM_VERSION_SUFFIX
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
    set(COMPILER_OPTIONS
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER})
    set(RUNTIME_DEP) # Don't set runtime dependencies
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
  endforeach()

  # Populate the passthrough variables
  foreach(variableName ${CLANG_BOOTSTRAP_PASSTHROUGH} ${BOOTSTRAP_DEFAULT_PASSTHROUGH})
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
    ${cmake_3_1_EXCLUDE_FROM_ALL}
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
    ${cmake_3_4_USES_TERMINAL_OPTIONS}
    )

  # exclude really-install from main target
  set_target_properties(${NEXT_CLANG_STAGE} PROPERTIES _EP_really-install_EXCLUDE_FROM_MAIN On)
  ExternalProject_Add_Step(${NEXT_CLANG_STAGE} really-install
    COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
    COMMENT "Performing install step for '${NEXT_CLANG_STAGE}'"
    DEPENDEES build
    ${cmake_3_4_USES_TERMINAL}
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
      ${cmake_3_4_USES_TERMINAL}
    )

    if(target MATCHES "^stage[0-9]*")
      add_custom_target(${target} DEPENDS ${NEXT_CLANG_STAGE}-${target})
    endif()

    ExternalProject_Add_StepTargets(${NEXT_CLANG_STAGE} ${target})
  endforeach()
endif()
