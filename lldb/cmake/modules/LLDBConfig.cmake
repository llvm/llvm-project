include(CheckCXXSymbolExists)
include(CheckTypeSize)

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

set(default_disable_python OFF)
set(default_disable_curses OFF)
set(default_disable_libedit OFF)

if(DEFINED LLVM_ENABLE_LIBEDIT AND NOT LLVM_ENABLE_LIBEDIT)
  set(default_disable_libedit ON)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(default_disable_curses ON)
  set(default_disable_libedit ON)
elseif(CMAKE_SYSTEM_NAME MATCHES "Android")
  set(default_disable_python ON)
  set(default_disable_curses ON)
  set(default_disable_libedit ON)
elseif(IOS)
  set(default_disable_python ON)
endif()

option(LLDB_DISABLE_PYTHON "Disable Python scripting integration." ${default_disable_python})
option(LLDB_DISABLE_CURSES "Disable Curses integration." ${default_disable_curses})
option(LLDB_DISABLE_LIBEDIT "Disable the use of editline." ${default_disable_libedit})
option(LLDB_RELOCATABLE_PYTHON "Use the PYTHONHOME environment variable to locate Python." OFF)
option(LLDB_USE_SYSTEM_SIX "Use six.py shipped with system and do not install a copy of it" OFF)
option(LLDB_USE_ENTITLEMENTS "When codesigning, use entitlements if available" ON)
option(LLDB_BUILD_FRAMEWORK "Build LLDB.framework (Darwin only)" OFF)
option(LLDB_NO_INSTALL_DEFAULT_RPATH "Disable default RPATH settings in binaries" OFF)

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

  # Set designated directory for all dSYMs. Essentially, this emits the
  # framework's dSYM outside of the framework directory.
  if(LLVM_EXTERNALIZE_DEBUGINFO)
    set(LLVM_EXTERNALIZE_DEBUGINFO_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin CACHE STRING
        "Directory to emit dSYM files stripped from executables and libraries (Darwin Only)")
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

if (LLDB_DISABLE_CURSES)
  add_definitions( -DLLDB_DISABLE_CURSES )
endif()

if (LLDB_DISABLE_LIBEDIT)
  add_definitions( -DLLDB_DISABLE_LIBEDIT )
else()
  find_package(LibEdit REQUIRED)

  # Check if we libedit capable of handling wide characters (built with
  # '--enable-widec').
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


# On Windows, we can't use the normal FindPythonLibs module that comes with CMake,
# for a number of reasons.
# 1) Prior to MSVC 2015, it is only possible to embed Python if python itself was
#    compiled with an identical version (and build configuration) of MSVC as LLDB.
#    The standard algorithm does not take into account the differences between
#    a binary release distribution of python and a custom built distribution.
# 2) From MSVC 2015 and onwards, it is only possible to use Python 3.5 or later.
# 3) FindPythonLibs queries the registry to locate Python, and when looking for a
#    64-bit version of Python, since cmake.exe is a 32-bit executable, it will see
#    a 32-bit view of the registry.  As such, it is impossible for FindPythonLibs to
#    locate 64-bit Python libraries.
# This function is designed to address those limitations.  Currently it only partially
# addresses them, but it can be improved and extended on an as-needed basis.
function(find_python_libs_windows)
  if ("${PYTHON_HOME}" STREQUAL "")
    message("LLDB embedded Python on Windows requires specifying a value for PYTHON_HOME.  Python support disabled.")
    set(LLDB_DISABLE_PYTHON 1 PARENT_SCOPE)
    return()
  endif()

  file(TO_CMAKE_PATH "${PYTHON_HOME}/Include" PYTHON_INCLUDE_DIR)

  if(EXISTS "${PYTHON_INCLUDE_DIR}/patchlevel.h")
    file(STRINGS "${PYTHON_INCLUDE_DIR}/patchlevel.h" python_version_str
         REGEX "^#define[ \t]+PY_VERSION[ \t]+\"[^\"]+\"")
    string(REGEX REPLACE "^#define[ \t]+PY_VERSION[ \t]+\"([^\"+]+)[+]?\".*" "\\1"
         PYTHONLIBS_VERSION_STRING "${python_version_str}")
    message("-- Found Python version ${PYTHONLIBS_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)[.]([0-9]+)[.][0-9]+" "python\\1\\2" PYTHONLIBS_BASE_NAME "${PYTHONLIBS_VERSION_STRING}")
    unset(python_version_str)
  else()
    message("Unable to find ${PYTHON_INCLUDE_DIR}/patchlevel.h, Python installation is corrupt.")
    message("Python support will be disabled for this build.")
    set(LLDB_DISABLE_PYTHON 1 PARENT_SCOPE)
    return()
  endif()

  file(TO_CMAKE_PATH "${PYTHON_HOME}" PYTHON_HOME)
  # TODO(compnerd) when CMake Policy `CMP0091` is set to NEW, we should use
  # if(CMAKE_MSVC_RUNTIME_LIBRARY MATCHES MultiThreadedDebug)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    file(TO_CMAKE_PATH "${PYTHON_HOME}/python_d.exe" PYTHON_EXE)
    file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/${PYTHONLIBS_BASE_NAME}_d.lib" PYTHON_LIB)
    file(TO_CMAKE_PATH "${PYTHON_HOME}/${PYTHONLIBS_BASE_NAME}_d.dll" PYTHON_DLL)
  else()
    file(TO_CMAKE_PATH "${PYTHON_HOME}/python.exe" PYTHON_EXE)
    file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/${PYTHONLIBS_BASE_NAME}.lib" PYTHON_LIB)
    file(TO_CMAKE_PATH "${PYTHON_HOME}/${PYTHONLIBS_BASE_NAME}.dll" PYTHON_DLL)
  endif()

  foreach(component PYTHON_EXE;PYTHON_LIB;PYTHON_DLL)
    if(NOT EXISTS ${${component}})
      message("unable to find ${component}")
      unset(${component})
    endif()
  endforeach()

  if (NOT PYTHON_EXE OR NOT PYTHON_LIB OR NOT PYTHON_DLL)
    message("Unable to find all Python components.  Python support will be disabled for this build.")
    set(LLDB_DISABLE_PYTHON 1 PARENT_SCOPE)
    return()
  endif()

  set (PYTHON_EXECUTABLE ${PYTHON_EXE} PARENT_SCOPE)
  set (PYTHON_LIBRARY ${PYTHON_LIB} PARENT_SCOPE)
  set (PYTHON_DLL ${PYTHON_DLL} PARENT_SCOPE)
  set (PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR} PARENT_SCOPE)

  message("-- LLDB Found PythonExecutable: ${PYTHON_EXE}")
  message("-- LLDB Found PythonLibs: ${PYTHON_LIB}")
  message("-- LLDB Found PythonDLL: ${PYTHON_DLL}")
  message("-- LLDB Found PythonIncludeDirs: ${PYTHON_INCLUDE_DIR}")
endfunction(find_python_libs_windows)

if (NOT LLDB_DISABLE_PYTHON)

  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    find_python_libs_windows()

    if (NOT LLDB_RELOCATABLE_PYTHON)
      file(TO_CMAKE_PATH "${PYTHON_HOME}" LLDB_PYTHON_HOME)
      add_definitions( -DLLDB_PYTHON_HOME="${LLDB_PYTHON_HOME}" )
    endif()
  else()
    find_package(PythonInterp)
    find_package(PythonLibs)
  endif()

  if (PYTHON_INCLUDE_DIR)
    include_directories(${PYTHON_INCLUDE_DIR})
  endif()
endif()

if (LLDB_DISABLE_PYTHON)
  unset(PYTHON_INCLUDE_DIR)
  unset(PYTHON_LIBRARY)
  unset(PYTHON_EXECUTABLE)
  add_definitions( -DLLDB_DISABLE_PYTHON )
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

check_cxx_compiler_flag("-Wno-gnu-anonymous-struct"
                        CXX_SUPPORTS_NO_GNU_ANONYMOUS_STRUCT)

check_cxx_compiler_flag("-Wno-nested-anon-types"
                        CXX_SUPPORTS_NO_NESTED_ANON_TYPES)

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

include_directories(BEFORE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/
    COMPONENT lldb-headers
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN ".svn" EXCLUDE
    PATTERN ".cmake" EXCLUDE
    PATTERN "Config.h" EXCLUDE
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
    find_library(DEBUG_SYMBOLS_LIBRARY DebugSymbols PATHS "/System/Library/PrivateFrameworks")
  endif()
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(SECURITY_LIBRARY Security)

  add_definitions( -DLIBXML2_DEFINED )
  list(APPEND system_libs xml2
       ${CURSES_LIBRARIES}
       ${FOUNDATION_LIBRARY}
       ${CORE_FOUNDATION_LIBRARY}
       ${CORE_SERVICES_LIBRARY}
       ${SECURITY_LIBRARY}
       ${DEBUG_SYMBOLS_LIBRARY})
  include_directories(${LIBXML2_INCLUDE_DIR})
elseif(LIBXML2_FOUND AND LIBXML2_VERSION_STRING VERSION_GREATER 2.8)
  add_definitions( -DLIBXML2_DEFINED )
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
if (CMAKE_SYSTEM_NAME MATCHES "Android|Darwin|FreeBSD|Linux|NetBSD")
  set(LLDB_CAN_USE_LLDB_SERVER 1)
else()
  set(LLDB_CAN_USE_LLDB_SERVER 0)
endif()

# Figure out if lldb could use debugserver.  If so, then we'll
# ensure we build debugserver when we build lldb.
if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
    set(LLDB_CAN_USE_DEBUGSERVER 1)
else()
    set(LLDB_CAN_USE_DEBUGSERVER 0)
endif()

if (NOT LLDB_DISABLE_CURSES)
    find_package(Curses REQUIRED)

    find_library(CURSES_PANEL_LIBRARY NAMES panel DOC "The curses panel library")
    if (NOT CURSES_PANEL_LIBRARY)
        message(FATAL_ERROR "A required curses' panel library not found.")
    endif ()

    # Add panels to the library path
    set (CURSES_LIBRARIES ${CURSES_LIBRARIES} ${CURSES_PANEL_LIBRARY})

    list(APPEND system_libs ${CURSES_LIBRARIES})
    include_directories(${CURSES_INCLUDE_DIR})
endif ()

check_cxx_symbol_exists("__GLIBCXX__" "string" LLDB_USING_LIBSTDCXX)
if(LLDB_USING_LIBSTDCXX)
    # There doesn't seem to be an easy way to check the library version. Instead, we rely on the
    # fact that std::set did not have the allocator constructor available until version 4.9
    check_cxx_source_compiles("
            #include <set>
            std::set<int> s = std::set<int>(std::allocator<int>());
            int main() { return 0; }"
            LLDB_USING_LIBSTDCXX_4_9)
    if (NOT LLDB_USING_LIBSTDCXX_4_9 AND NOT LLVM_ENABLE_EH)
        message(WARNING
            "You appear to be linking to libstdc++ version lesser than 4.9 without exceptions "
            "enabled. These versions of the library have an issue, which causes occasional "
            "lldb crashes. See <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59656> for "
            "details. Possible courses of action are:\n"
            "- use libstdc++ version 4.9 or newer\n"
            "- use libc++ (via LLVM_ENABLE_LIBCXX)\n"
            "- enable exceptions (via LLVM_ENABLE_EH)\n"
            "- ignore this warning and accept occasional instability")
    endif()
endif()

if ((CMAKE_SYSTEM_NAME MATCHES "Android") AND LLVM_BUILD_STATIC AND
    ((ANDROID_ABI MATCHES "armeabi") OR (ANDROID_ABI MATCHES "mips")))
  add_definitions(-DANDROID_USE_ACCEPT_WORKAROUND)
endif()

find_package(Backtrace)
include(LLDBGenerateConfig)
