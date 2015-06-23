if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  set(LLDB_DEFAULT_DISABLE_PYTHON 0)
  set(LLDB_DEFAULT_DISABLE_CURSES 1)
else()
  if ( __ANDROID_NDK__ )
    set(LLDB_DEFAULT_DISABLE_PYTHON 1)
    set(LLDB_DEFAULT_DISABLE_CURSES 1)
  else()
    set(LLDB_DEFAULT_DISABLE_PYTHON 0)
    set(LLDB_DEFAULT_DISABLE_CURSES 0)
  endif()
endif()
set(LLDB_DISABLE_PYTHON ${LLDB_DEFAULT_DISABLE_PYTHON} CACHE BOOL
  "Disables the Python scripting integration.")
set(LLDB_DISABLE_CURSES ${LLDB_DEFAULT_DISABLE_CURSES} CACHE BOOL
  "Disables the Curses integration.")

if ( LLDB_DISABLE_PYTHON )
  set(LLDB_DEFAULT_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION 0)
else ()
  set(LLDB_DEFAULT_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION 1)
endif ()

set(LLDB_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION ${LLDB_DEFAULT_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION} CACHE BOOL
  "Enables using new Python scripts for SWIG API generation .")

# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)
  cmake_minimum_required(VERSION 2.8)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY "Only include toolchain files in the 'install' target." OFF)

  set(LLDB_PATH_TO_LLVM_SOURCE "" CACHE PATH
    "Path to LLVM source code. Not necessary if using an installed LLVM.")
  set(LLDB_PATH_TO_LLVM_BUILD "" CACHE PATH
    "Path to the directory where LLVM was built or installed.")

  set(LLDB_PATH_TO_CLANG_SOURCE "" CACHE PATH
    "Path to Clang source code. Not necessary if using an installed Clang.")
  set(LLDB_PATH_TO_CLANG_BUILD "" CACHE PATH
    "Path to the directory where Clang was built or installed.")

  if (LLDB_PATH_TO_LLVM_SOURCE)
    if (NOT EXISTS "${LLDB_PATH_TO_LLVM_SOURCE}/cmake/config-ix.cmake")
      message(FATAL_ERROR "Please set LLDB_PATH_TO_LLVM_SOURCE to the root "
              "directory of LLVM source code.")
    else()
      get_filename_component(LLVM_MAIN_SRC_DIR ${LLDB_PATH_TO_LLVM_SOURCE}
                             ABSOLUTE)
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  if (LLDB_PATH_TO_CLANG_SOURCE)
      get_filename_component(CLANG_MAIN_SRC_DIR ${LLDB_PATH_TO_CLANG_SOURCE}
                             ABSOLUTE)
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  get_filename_component(PATH_TO_LLVM_BUILD ${LLDB_PATH_TO_LLVM_BUILD}
                         ABSOLUTE)

  get_filename_component(PATH_TO_CLANG_BUILD ${LLDB_PATH_TO_CLANG_BUILD}
                         ABSOLUTE)

  # These variables are used by add_llvm_library.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  include(AddLLVM)
  include(HandleLLVMOptions)

  # Verify that we can find a Python 2 interpreter.  Python 3 is unsupported.
  set(Python_ADDITIONAL_VERSIONS 2.7 2.6 2.5)
  include(FindPythonInterp)
  if( NOT PYTHONINTERP_FOUND )
    message(FATAL_ERROR
  "Unable to find Python interpreter, required for builds and testing.

  Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
  endif()

  # Import CMake library targets from LLVM and Clang.
  include("${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  if ( EXISTS "${LLDB_PATH_TO_CLANG_BUILD}/share/clang/cmake/ClangConfig.cmake" )
    include("${LLDB_PATH_TO_CLANG_BUILD}/share/clang/cmake/ClangConfig.cmake")
  endif()

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CLANG_MAIN_INCLUDE_DIR "${CLANG_MAIN_SRC_DIR}/include")

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${PATH_TO_LLVM_BUILD}/include"
                      "${LLVM_MAIN_INCLUDE_DIR}"
                      "${PATH_TO_CLANG_BUILD}/include"
                      "${CLANG_MAIN_INCLUDE_DIR}"
                      "${CMAKE_CURRENT_SOURCE_DIR}/source")
  link_directories("${PATH_TO_LLVM_BUILD}/lib${LLVM_LIBDIR_SUFFIX}"
                   "${PATH_TO_CLANG_BUILD}/lib${LLVM_LIBDIR_SUFFIX}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(LLDB_BUILT_STANDALONE 1)
endif()

set(LLDB_DISABLE_PYTHON 0 CACHE BOOL "Disables the Python scripting integration.")
if (LLDB_DISABLE_PYTHON)
  add_definitions( -DLLDB_DISABLE_PYTHON )
endif()

if ((NOT MSVC) OR MSVC12)
  add_definitions( -DHAVE_ROUND )
endif()

if (LLDB_DISABLE_CURSES)
  add_definitions( -DLLDB_DISABLE_CURSES )
endif()

macro(add_lldb_definitions)
  # We don't want no semicolons on LLDB_DEFINITIONS:
  foreach(arg ${ARGN})
    set(LLDB_DEFINITIONS "${LLVM_DEFINITIONS} ${arg}")
  endforeach(arg)
  add_definitions( ${ARGN} )
endmacro(add_lldb_definitions)

if (NOT LLDB_DISABLE_PYTHON)
  if(UNIX)
    # This is necessary for crosscompile on Ubuntu 14.04 64bit. Need a proper fix.
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
    endif()
  endif()
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})
endif()

include_directories(../clang/include)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../clang/include")

# lldb requires c++11 to build. Make sure that we have a compiler and standard
# library combination that can do that.
if (NOT MSVC)
  # gcc and clang require the -std=c++0x or -std=c++11 flag.
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (NOT ("${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+0x" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+11" OR
             "${CMAKE_CXX_FLAGS}" MATCHES "-std=gnu\\+\\+11"))
      if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.7")
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        else()
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
        endif()
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      endif()
    endif()
  endif()
elseif (MSVC_VERSION LESS 1700)
  message(FATAL_ERROR "The selected compiler does not support c++11 which is "
          "required to build lldb.")
endif()

# Disable GCC warnings
check_cxx_compiler_flag("-Wno-deprecated-declarations"
                        CXX_SUPPORTS_NO_DEPRECATED_DECLARATIONS)
if (CXX_SUPPORTS_NO_DEPRECATED_DECLARATIONS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif ()

check_cxx_compiler_flag("-Wno-unknown-pragmas"
                        CXX_SUPPORTS_NO_UNKNOWN_PRAGMAS)
if (CXX_SUPPORTS_NO_UNKNOWN_PRAGMAS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

# Disable Clang warnings
check_cxx_compiler_flag("-Wno-deprecated-register"
                        CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
if (CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif ()

# Disable MSVC warnings
if( MSVC )
  add_lldb_definitions(
    -wd4018 # Suppress 'warning C4018: '>=' : signed/unsigned mismatch'
    -wd4068 # Suppress 'warning C4068: unknown pragma'
    -wd4150 # Suppress 'warning C4150: deletion of pointer to incomplete type'
    -wd4251 # Suppress 'warning C4251: T must have dll-interface to be used by clients of class U.'
    -wd4521 # Suppress 'warning C4521: 'type' : multiple copy constructors specified'
    -wd4530 # Suppress 'warning C4530: C++ exception handler used, but unwind semantics are not enabled.'
  )
endif()

# If building on a 32-bit system, make sure off_t can store offsets > 2GB
if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
  add_lldb_definitions(
    -D_LARGEFILE_SOURCE
    -D_FILE_OFFSET_BITS=64
    )
endif()

set(LLDB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. CMake would overwrite "
"the makefiles distributed with LLDB. Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

# Compute the LLDB version from the LLVM version.
string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" LLDB_VERSION
  ${PACKAGE_VERSION})
message(STATUS "LLDB version: ${LLDB_VERSION}")

if (CMAKE_VERSION VERSION_LESS 2.8.12)
  set(cmake_2_8_12_INTERFACE)
  set(cmake_2_8_12_PRIVATE)
  set(cmake_2_8_12_PUBLIC)
else ()
  set(cmake_2_8_12_INTERFACE INTERFACE)
  set(cmake_2_8_12_PRIVATE PRIVATE)
  set(cmake_2_8_12_PUBLIC PUBLIC)
endif ()

macro(add_lldb_library name)
  llvm_process_sources(srcs ${ARGN})
  if (MSVC_IDE OR XCODE)
    string(REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list(GET split_path -1 dir)
    file(GLOB_RECURSE headers
      ../../include/lldb${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif()
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind STATIC)
  endif()
  #PIC not needed on Win
  if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif()
  llvm_add_library(${name} ${libkind} ${srcs})
  #if (LLVM_COMMON_DEPENDS)
  ##add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  #endif()

  if ("${libkind}" STREQUAL "STATIC")
    set(lldb_library_keyword ${cmake_2_8_12_INTERFACE})
  else ()
    set(lldb_library_keyword ${cmake_2_8_12_PUBLIC})
  endif ()

  if(LLDB_USED_LIBS)
    # The Darwin linker doesn't understand --start-group/--end-group.
    if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
      target_link_libraries(${name} ${lldb_library_keyword}
                            -Wl,--start-group ${LLDB_USED_LIBS} -Wl,--end-group)
    else()
      target_link_libraries(${name} ${lldb_library_keyword} ${LLDB_USED_LIBS})
    endif()
  endif()
  target_link_libraries(${name} ${lldb_library_keyword} ${CLANG_USED_LIBS})
  target_link_libraries(${name} ${lldb_library_keyword} ${LLVM_USED_LIBS})
  llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  target_link_libraries(${name} ${lldb_library_keyword} ${LLVM_COMMON_LIBS})
  if (LLVM_COMMON_DEPENDS)
    add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
  endif()

  # Hack: only some LLDB libraries depend on the clang autogenerated headers,
  # but it is simple enough to make all of LLDB depend on some of those
  # headers without negatively impacting much of anything.
  set (LLDB_DEPENDENCIES
    libclang
    )
  add_dependencies(${name} ${LLDB_DEPENDENCIES})

  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "liblldb")
    install(TARGETS ${name}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "lldb libraries")
endmacro(add_lldb_library)

macro(add_lldb_executable name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "lldb executables")
endmacro(add_lldb_executable)

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


# Find libraries or frameworks that may be needed
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_library(CARBON_LIBRARY Carbon)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(CORE_SERVICES_LIBRARY CoreServices)
  find_library(SECURITY_LIBRARY Security)
  find_library(DEBUG_SYMBOLS_LIBRARY DebugSymbols PATHS "/System/Library/PrivateFrameworks")

  if (NOT LIBXML2_FOUND)
    find_package(LibXml2)
  endif ()
  list(APPEND system_libs xml2 ncurses panel)
  list(APPEND system_libs ${CARBON_LIBRARY} ${FOUNDATION_LIBRARY}
  ${CORE_FOUNDATION_LIBRARY} ${CORE_SERVICES_LIBRARY} ${SECURITY_LIBRARY}
  ${DEBUG_SYMBOLS_LIBRARY})
endif()

if(LLDB_REQUIRES_EH)
  set(LLDB_REQUIRES_RTTI ON)
else()
  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    set(LLDB_COMPILE_FLAGS "${LLDB_COMPILE_FLAGS} -fno-exceptions")
  elseif(MSVC)
    add_definitions( -D_HAS_EXCEPTIONS=0 )
    set(LLDB_COMPILE_FLAGS "${LLDB_COMPILE_FLAGS} /EHs-c-")
  endif()
endif()

# Disable RTTI by default
if(NOT LLDB_REQUIRES_RTTI)
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    set(LLDB_COMPILE_FLAGS "${LLDB_COMPILE_FLAGS} -fno-rtti")
  elseif(MSVC)
    set(LLDB_COMPILE_FLAGS "${LLDB_COMPILE_FLAGS} /GR-")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLDB_COMPILE_FLAGS}")

#add_subdirectory(include)
add_subdirectory(docs)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" OR NOT LLDB_DISABLE_PYTHON)
  add_subdirectory(scripts)
endif ()
add_subdirectory(source)
add_subdirectory(test)
add_subdirectory(tools)
