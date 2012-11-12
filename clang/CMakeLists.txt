# If we are not building as a part of LLVM, build Clang as an
# standalone project, using LLVM as an external library:
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(Clang)
  cmake_minimum_required(VERSION 2.8)

  set(CLANG_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(CLANG_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

  if( CLANG_PATH_TO_LLVM_SOURCE )
    if( NOT EXISTS "${CLANG_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake" )
      message(FATAL_ERROR "Please set CLANG_PATH_TO_LLVM_SOURCE to the root directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${CLANG_PATH_TO_LLVM_SOURCE}
	ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  if( NOT EXISTS "${CLANG_PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    # Looking for bin/Debug/llvm-tblgen is a complete hack. How can we get
    # around this?
    if( NOT EXISTS "${CLANG_PATH_TO_LLVM_BUILD}/bin/Debug/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
      message(FATAL_ERROR "Please set CLANG_PATH_TO_LLVM_BUILD to a directory containing a LLVM build.")
    endif()
  endif()

  list(APPEND CMAKE_MODULE_PATH "${CLANG_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${CLANG_PATH_TO_LLVM_BUILD}
    ABSOLUTE)

  include(AddLLVM)
  include(TableGen)
  include("${CLANG_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  include(HandleLLVMOptions)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${PATH_TO_LLVM_BUILD}/lib")

  if( EXISTS "${CLANG_PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}" )
    set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    # FIXME: This is an utter hack.
    set(LLVM_TABLEGEN_EXE "${PATH_TO_LLVM_BUILD}/bin/Debug/llvm-tblgen${CMAKE_EXECUTABLE_SUFFIX}")
  endif()

  # Define the default arguments to use with 'lit', and an option for the user
  # to override.
  set(LIT_ARGS_DEFAULT "-sv")
  if (MSVC OR XCODE)
    set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
  endif()
  set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

  set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
  set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

  set( CLANG_BUILT_STANDALONE 1 )
endif()

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed." )
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>." )

set(CLANG_VENDOR "" CACHE STRING
  "Vendor-specific text for showing with version information.")

if( CLANG_VENDOR )
  add_definitions( -DCLANG_VENDOR="${CLANG_VENDOR} " )
endif()

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings")
endif ()

if (APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif ()

# libxml2 is an optional dependency, required only to run validation
# tests on XML output.
find_package(LibXml2)

configure_file(
  ${CLANG_SOURCE_DIR}/include/clang/Config/config.h.cmake
  ${CLANG_BINARY_DIR}/include/clang/Config/config.h)

include(LLVMParseArguments)

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

  parse_arguments( CTG "SOURCE;TARGET;DEPENDS" "" ${ARGN} )

  if( NOT CTG_SOURCE )
    message(FATAL_ERROR "SOURCE source-file required by clang_tablegen")
  endif()

  set( LLVM_TARGET_DEFINITIONS ${CTG_SOURCE} )
  tablegen( CLANG ${CTG_DEFAULT_ARGS} )

  list( GET CTG_DEFAULT_ARGS 0 output_file )
  if( CTG_TARGET )
    add_custom_target( ${CTG_TARGET} DEPENDS ${output_file} ${CTG_DEPENDS} )
    set_target_properties( ${CTG_TARGET} PROPERTIES FOLDER "Clang tablegenning")
  endif()
endfunction(clang_tablegen)

macro(add_clang_library name)
  llvm_process_sources(srcs ${ARGN})
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

      set(srcs ${srcs} ${headers} ${tds})
    endif()
  endif(MSVC_IDE OR XCODE)
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library( ${name} ${libkind} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  llvm_config( ${name} ${LLVM_LINK_COMPONENTS} )
  target_link_libraries( ${name} ${LLVM_COMMON_LIBS} )
  link_system_libs( ${name} )

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION bin)
  set_target_properties(${name} PROPERTIES FOLDER "Clang libraries")
endmacro(add_clang_library)

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "Clang executables")
endmacro(add_clang_executable)

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "config.h" EXCLUDE
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "*.inc"
  )

add_definitions( -D_GNU_SOURCE )

# Clang version information
set(CLANG_EXECUTABLE_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the clang executable, in the form XX.YY")
set(LIBCLANG_LIBRARY_VERSION
     "${CLANG_VERSION_MAJOR}.${CLANG_VERSION_MINOR}" CACHE STRING
    "Version number that will be placed into the libclang library , in the form XX.YY")
mark_as_advanced(CLANG_EXECUTABLE_VERSION LIBCLANG_LIBRARY_VERSION)

add_subdirectory(utils/TableGen)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

option(CLANG_BUILD_EXAMPLES "Build CLANG example programs by default." OFF)
add_subdirectory(examples)

# TODO: docs.
add_subdirectory(test)

if( LLVM_INCLUDE_TESTS )
  if( NOT CLANG_BUILT_STANDALONE )
    add_subdirectory(unittests)
  endif()
endif()

# Workaround for MSVS10 to avoid the Dialog Hell
# FIXME: This could be removed with future version of CMake.
if( CLANG_BUILT_STANDALONE AND MSVC_VERSION EQUAL 1600 )
  set(CLANG_SLN_FILENAME "${CMAKE_CURRENT_BINARY_DIR}/Clang.sln")
  if( EXISTS "${CLANG_SLN_FILENAME}" )
    file(APPEND "${CLANG_SLN_FILENAME}" "\n# This should be regenerated!\n")
  endif()
endif()

set(BUG_REPORT_URL "http://llvm.org/bugs/" CACHE STRING
  "Default URL where bug reports are to be submitted.")

