include(CheckCXXSymbolExists)
include(CheckTypeSize)
include(CMakeDependentOption)

set(LLDB_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(LLDB_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LLDB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. CMake would overwrite the makefiles "
    "distributed with LLDB. Please create a directory and run cmake from "
    "there, passing the path to this source directory as the last argument. "
    "This process created the file `CMakeCache.txt' and the directory "
    "`CMakeFiles'. Please delete them.")
endif()

set(LLDB_LINKER_SUPPORTS_GROUPS OFF)
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  # The Darwin linker doesn't understand --start-group/--end-group.
  set(LLDB_LINKER_SUPPORTS_GROUPS ON)
endif()

macro(add_optional_dependency variable description package found)
  set(${variable} "Auto" CACHE STRING "${description} On, Off or Auto (default)")
  string(TOUPPER "${${variable}}" ${variable})

  if("${${variable}}" STREQUAL "AUTO")
    set(find_package TRUE)
    set(maybe_required)
  elseif(${${variable}})
    set(find_package TRUE)
    set(maybe_required REQUIRED)
  else()
    set(find_package FALSE)
    set(${variable} FALSE PARENT_SCOPE)
  endif()

  if(${find_package})
    find_package(${package} ${maybe_required})
    set(${variable} "${${found}}")
  endif()
endmacro()

add_optional_dependency(LLDB_ENABLE_LIBEDIT "Enable editline support." LibEdit libedit_FOUND)
add_optional_dependency(LLDB_ENABLE_CURSES "Enable curses support." CursesAndPanel CURSESANDPANEL_FOUND)
add_optional_dependency(LLDB_ENABLE_LZMA "Enable LZMA compression support." LibLZMA LIBLZMA_FOUND)
add_optional_dependency(LLDB_ENABLE_LUA "Enable Lua scripting support." Lua LUA_FOUND)

set(default_enable_python ON)

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  set(default_enable_python OFF)
elseif(IOS)
  set(default_enable_python OFF)
endif()

option(LLDB_ENABLE_PYTHON "Enable Python scripting integration." ${default_enable_python})
option(LLDB_RELOCATABLE_PYTHON "Use the PYTHONHOME environment variable to locate Python." OFF)
option(LLDB_USE_SYSTEM_SIX "Use six.py shipped with system and do not install a copy of it" OFF)
option(LLDB_USE_ENTITLEMENTS "When codesigning, use entitlements if available" ON)
option(LLDB_BUILD_FRAMEWORK "Build LLDB.framework (Darwin only)" OFF)
option(LLDB_NO_INSTALL_DEFAULT_RPATH "Disable default RPATH settings in binaries" OFF)
option(LLDB_USE_SYSTEM_DEBUGSERVER "Use the system's debugserver for testing (Darwin only)." OFF)
option(LLDB_SKIP_STRIP "Whether to skip stripping of binaries when installing lldb." OFF)

if (LLDB_USE_SYSTEM_DEBUGSERVER)
  # The custom target for the system debugserver has no install target, so we
  # need to remove it from the LLVM_DISTRIBUTION_COMPONENTS list.
  if (LLVM_DISTRIBUTION_COMPONENTS)
    list(REMOVE_ITEM LLVM_DISTRIBUTION_COMPONENTS debugserver)
    set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS} CACHE STRING "" FORCE)
  endif()
endif()

if(LLDB_BUILD_FRAMEWORK)
  if(NOT APPLE)
    message(FATAL_ERROR "LLDB.framework can only be generated when targeting Apple platforms")
  endif()
  # CMake 3.6 did not correctly emit POST_BUILD commands for Apple Framework targets
  # CMake < 3.8 did not have the BUILD_RPATH target property
  if(CMAKE_VERSION VERSION_LESS 3.8)
    message(FATAL_ERROR "LLDB_BUILD_FRAMEWORK is not supported on CMake < 3.8")
  endif()

  set(LLDB_FRAMEWORK_VERSION A CACHE STRING "LLDB.framework version (default is A)")
  set(LLDB_FRAMEWORK_BUILD_DIR bin CACHE STRING "Output directory for LLDB.framework")
  set(LLDB_FRAMEWORK_INSTALL_DIR Library/Frameworks CACHE STRING "Install directory for LLDB.framework")

  get_filename_component(LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR ${LLDB_FRAMEWORK_BUILD_DIR} ABSOLUTE
    BASE_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR})

  # Essentially, emit the framework's dSYM outside of the framework directory.
  set(LLDB_DEBUGINFO_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin CACHE STRING
      "Directory to emit dSYM files stripped from executables and libraries (Darwin Only)")
endif()

if(APPLE AND CMAKE_GENERATOR STREQUAL Xcode)
  if(NOT LLDB_EXPLICIT_XCODE_CACHE_USED)
    message(WARNING
      "When building with Xcode, we recommend using the corresponding cache script. "
      "If this was a mistake, clean your build directory and re-run CMake with:\n"
      "  -C ${CMAKE_SOURCE_DIR}/cmake/caches/Apple-lldb-Xcode.cmake\n"
      "See: https://lldb.llvm.org/resources/build.html#cmakegeneratedxcodeproject\n")
  endif()
endif()

if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(LLDB_EXPORT_ALL_SYMBOLS 0 CACHE BOOL
    "Causes lldb to export all symbols when building liblldb.")
else()
  # Windows doesn't support toggling this, so don't bother making it a
  # cache variable.
  set(LLDB_EXPORT_ALL_SYMBOLS 0)
endif()

if ((NOT MSVC) OR MSVC12)
  add_definitions( -DHAVE_ROUND )
endif()

# Check if we libedit capable of handling wide characters (built with
# '--enable-widec').
if (LLDB_ENABLE_LIBEDIT)
  set(CMAKE_REQUIRED_LIBRARIES ${libedit_LIBRARIES})
  set(CMAKE_REQUIRED_INCLUDES ${libedit_INCLUDE_DIRS})
  check_symbol_exists(el_winsertstr histedit.h LLDB_EDITLINE_USE_WCHAR)
  set(CMAKE_EXTRA_INCLUDE_FILES histedit.h)
  check_type_size(el_rfunc_t LLDB_EL_RFUNC_T_SIZE)
  if (LLDB_EL_RFUNC_T_SIZE STREQUAL "")
    set(LLDB_HAVE_EL_RFUNC_T 0)
  else()
    set(LLDB_HAVE_EL_RFUNC_T 1)
  endif()
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_EXTRA_INCLUDE_FILES)
endif()

if (LLDB_ENABLE_PYTHON)
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(Python3_VERSION VERSION_LESS 3.5)
      message(SEND_ERROR "Python 3.5 or newer is required (found: ${Python3_VERSION}")
    endif()
    set(PYTHON_LIBRARY ${Python3_LIBRARIES})
    include_directories(${Python3_INCLUDE_DIRS})

    if (NOT LLDB_RELOCATABLE_PYTHON)
      get_filename_component(PYTHON_HOME "${Python3_EXECUTABLE}" DIRECTORY)
      file(TO_CMAKE_PATH "${PYTHON_HOME}" LLDB_PYTHON_HOME)
    endif()
  else()
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)

    if (NOT CMAKE_CROSSCOMPILING)
      string(REPLACE "." ";" pythonlibs_version_list ${PYTHONLIBS_VERSION_STRING})
      list(GET pythonlibs_version_list 0 pythonlibs_major)
      list(GET pythonlibs_version_list 1 pythonlibs_minor)

      # Ignore the patch version. Some versions of macOS report a different patch
      # version for the system provided interpreter and libraries.
      if (NOT PYTHON_VERSION_MAJOR VERSION_EQUAL pythonlibs_major OR
          NOT PYTHON_VERSION_MINOR VERSION_EQUAL pythonlibs_minor)
        message(FATAL_ERROR "Found incompatible Python interpreter (${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})"
                            " and Python libraries (${pythonlibs_major}.${pythonlibs_minor})")
      endif()
    endif()

    if (PYTHON_INCLUDE_DIR)
      include_directories(${PYTHON_INCLUDE_DIR})
    endif()
  endif()
endif()

if (NOT LLDB_ENABLE_PYTHON)
  unset(PYTHON_INCLUDE_DIR)
  unset(PYTHON_LIBRARY)
  unset(PYTHON_EXECUTABLE)
endif()

if (LLVM_EXTERNAL_CLANG_SOURCE_DIR)
  include_directories(${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include)
else ()
  include_directories(${CMAKE_SOURCE_DIR}/tools/clang/include)
endif ()
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../clang/include")

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

check_cxx_compiler_flag("-Wno-strict-aliasing"
                        CXX_SUPPORTS_NO_STRICT_ALIASING)
if (CXX_SUPPORTS_NO_STRICT_ALIASING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
endif ()

# Disable Clang warnings
check_cxx_compiler_flag("-Wno-deprecated-register"
                        CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
if (CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif ()

check_cxx_compiler_flag("-Wno-vla-extension"
                        CXX_SUPPORTS_NO_VLA_EXTENSION)
if (CXX_SUPPORTS_NO_VLA_EXTENSION)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla-extension")
endif ()

# Disable MSVC warnings
if( MSVC )
  add_definitions(
    -wd4018 # Suppress 'warning C4018: '>=' : signed/unsigned mismatch'
    -wd4068 # Suppress 'warning C4068: unknown pragma'
    -wd4150 # Suppress 'warning C4150: deletion of pointer to incomplete type'
    -wd4201 # Suppress 'warning C4201: nonstandard extension used: nameless struct/union'
    -wd4251 # Suppress 'warning C4251: T must have dll-interface to be used by clients of class U.'
    -wd4521 # Suppress 'warning C4521: 'type' : multiple copy constructors specified'
    -wd4530 # Suppress 'warning C4530: C++ exception handler used, but unwind semantics are not enabled.'
  )
endif()

# Use the Unicode (UTF-16) APIs by default on Win32
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_definitions( -D_UNICODE -DUNICODE )
endif()

# If LLDB_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED LLDB_VERSION_MAJOR)
  set(LLDB_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED LLDB_VERSION_MINOR)
  set(LLDB_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED LLDB_VERSION_PATCH)
  set(LLDB_VERSION_PATCH ${LLVM_VERSION_PATCH})
endif()
if(NOT DEFINED LLDB_VERSION_SUFFIX)
  set(LLDB_VERSION_SUFFIX ${LLVM_VERSION_SUFFIX})
endif()
set(LLDB_VERSION "${LLDB_VERSION_MAJOR}.${LLDB_VERSION_MINOR}.${LLDB_VERSION_PATCH}${LLDB_VERSION_SUFFIX}")
message(STATUS "LLDB version: ${LLDB_VERSION}")

if (LLDB_ENABLE_LZMA)
  include_directories(${LIBLZMA_INCLUDE_DIRS})
endif()

include_directories(BEFORE
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/
    COMPONENT lldb-headers
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
    PATTERN ".cmake" EXCLUDE
    )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
    COMPONENT lldb-headers
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
    PATTERN ".cmake" EXCLUDE
    )

  add_custom_target(lldb-headers)
  set_target_properties(lldb-headers PROPERTIES FOLDER "lldb misc")

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-lldb-headers
                             COMPONENT lldb-headers)
  endif()
endif()

if (NOT LIBXML2_FOUND)
  find_package(LibXml2)
endif()

# Find libraries or frameworks that may be needed
if (APPLE)
  if(NOT IOS)
    find_library(CARBON_LIBRARY Carbon)
    find_library(CORE_SERVICES_LIBRARY CoreServices)
  endif()
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(SECURITY_LIBRARY Security)
  set(LLDB_ENABLE_LIBXML2 ON)
  list(APPEND system_libs xml2
       ${FOUNDATION_LIBRARY}
       ${CORE_FOUNDATION_LIBRARY}
       ${CORE_SERVICES_LIBRARY}
       ${SECURITY_LIBRARY}
       ${DEBUG_SYMBOLS_LIBRARY})
  include_directories(${LIBXML2_INCLUDE_DIR})
elseif(LIBXML2_FOUND AND LIBXML2_VERSION_STRING VERSION_GREATER 2.8)
  set(LLDB_ENABLE_LIBXML2 ON)
  list(APPEND system_libs ${LIBXML2_LIBRARIES})
  include_directories(${LIBXML2_INCLUDE_DIR})
endif()

if( WIN32 AND NOT CYGWIN )
  set(PURE_WINDOWS 1)
endif()

if(NOT PURE_WINDOWS)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  find_package(Threads REQUIRED)
  list(APPEND system_libs ${CMAKE_THREAD_LIBS_INIT})
endif()

list(APPEND system_libs ${CMAKE_DL_LIBS})

# Figure out if lldb could use lldb-server.  If so, then we'll
# ensure we build lldb-server when an lldb target is being built.
if (CMAKE_SYSTEM_NAME MATCHES "Android|Darwin|FreeBSD|Linux|NetBSD|Windows")
  set(LLDB_CAN_USE_LLDB_SERVER ON)
else()
  set(LLDB_CAN_USE_LLDB_SERVER OFF)
endif()

# Figure out if lldb could use debugserver.  If so, then we'll
# ensure we build debugserver when we build lldb.
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(LLDB_CAN_USE_DEBUGSERVER ON)
else()
    set(LLDB_CAN_USE_DEBUGSERVER OFF)
endif()

if ((CMAKE_SYSTEM_NAME MATCHES "Android") AND LLVM_BUILD_STATIC AND
    ((ANDROID_ABI MATCHES "armeabi") OR (ANDROID_ABI MATCHES "mips")))
  add_definitions(-DANDROID_USE_ACCEPT_WORKAROUND)
endif()

find_package(Backtrace)
include(LLDBGenerateConfig)
