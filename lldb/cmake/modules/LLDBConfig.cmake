include(CheckCXXSymbolExists)

set(LLDB_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/source")
set(LLDB_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LLDB_LINKER_SUPPORTS_GROUPS OFF)
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  # The Darwin linker doesn't understand --start-group/--end-group.
  set(LLDB_LINKER_SUPPORTS_GROUPS ON)
endif()

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

set(LLDB_RELOCATABLE_PYTHON 0 CACHE BOOL
  "Causes LLDB to use the PYTHONHOME environment variable to locate Python.")

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

  file(TO_CMAKE_PATH "${PYTHON_HOME}/Include" PYTHON_INCLUDE_DIRS)

  if(EXISTS "${PYTHON_INCLUDE_DIRS}/patchlevel.h")
    file(STRINGS "${PYTHON_INCLUDE_DIRS}/patchlevel.h" python_version_str
         REGEX "^#define[ \t]+PY_VERSION[ \t]+\"[^\"]+\"")
    string(REGEX REPLACE "^#define[ \t]+PY_VERSION[ \t]+\"([^\"+]+)[+]?\".*" "\\1"
         PYTHONLIBS_VERSION_STRING "${python_version_str}")
    message("-- Found Python version ${PYTHONLIBS_VERSION_STRING}")
    string(REGEX REPLACE "([0-9]+)[.]([0-9]+)[.][0-9]+" "python\\1\\2" PYTHONLIBS_BASE_NAME "${PYTHONLIBS_VERSION_STRING}")
    unset(python_version_str)
  else()
    message("Unable to find ${PYTHON_INCLUDE_DIRS}/patchlevel.h, Python installation is corrupt.")
    message("Python support will be disabled for this build.")
    set(LLDB_DISABLE_PYTHON 1 PARENT_SCOPE)
    return()
  endif()

  file(TO_CMAKE_PATH "${PYTHON_HOME}" PYTHON_HOME)
  file(TO_CMAKE_PATH "${PYTHON_HOME}/python_d.exe" PYTHON_DEBUG_EXE)
  file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/${PYTHONLIBS_BASE_NAME}_d.lib" PYTHON_DEBUG_LIB)
  file(TO_CMAKE_PATH "${PYTHON_HOME}/${PYTHONLIBS_BASE_NAME}_d.dll" PYTHON_DEBUG_DLL)

  file(TO_CMAKE_PATH "${PYTHON_HOME}/python.exe" PYTHON_RELEASE_EXE)
  file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/${PYTHONLIBS_BASE_NAME}.lib" PYTHON_RELEASE_LIB)
  file(TO_CMAKE_PATH "${PYTHON_HOME}/${PYTHONLIBS_BASE_NAME}.dll" PYTHON_RELEASE_DLL)

  if (NOT EXISTS ${PYTHON_DEBUG_EXE})
    message("Unable to find ${PYTHON_DEBUG_EXE}")
    unset(PYTHON_DEBUG_EXE)
  endif()

  if (NOT EXISTS ${PYTHON_RELEASE_EXE})
    message("Unable to find ${PYTHON_RELEASE_EXE}")
    unset(PYTHON_RELEASE_EXE)
  endif()

  if (NOT EXISTS ${PYTHON_DEBUG_LIB})
    message("Unable to find ${PYTHON_DEBUG_LIB}")
    unset(PYTHON_DEBUG_LIB)
  endif()

  if (NOT EXISTS ${PYTHON_RELEASE_LIB})
    message("Unable to find ${PYTHON_RELEASE_LIB}")
    unset(PYTHON_RELEASE_LIB)
  endif()

  if (NOT EXISTS ${PYTHON_DEBUG_DLL})
    message("Unable to find ${PYTHON_DEBUG_DLL}")
    unset(PYTHON_DEBUG_DLL)
  endif()

  if (NOT EXISTS ${PYTHON_RELEASE_DLL})
    message("Unable to find ${PYTHON_RELEASE_DLL}")
    unset(PYTHON_RELEASE_DLL)
  endif()

  if (NOT (PYTHON_DEBUG_EXE AND PYTHON_RELEASE_EXE AND PYTHON_DEBUG_LIB AND PYTHON_RELEASE_LIB AND PYTHON_DEBUG_DLL AND PYTHON_RELEASE_DLL))
    message("Python installation is corrupt. Python support will be disabled for this build.")
    set(LLDB_DISABLE_PYTHON 1 PARENT_SCOPE)
    return()
  endif()

  # Generator expressions are evaluated in the context of each build configuration generated
  # by CMake. Here we use the $<CONFIG:Debug>:VALUE logical generator expression to ensure
  # that the debug Python library, DLL, and executable are used in the Debug build configuration.
  #
  # Generator expressions can be difficult to grok at first so here's a breakdown of the one
  # used for PYTHON_LIBRARY:
  #
  # 1. $<CONFIG:Debug> evaluates to 1 when the Debug configuration is being generated,
  #    or 0 in all other cases.
  # 2. $<$<CONFIG:Debug>:${PYTHON_DEBUG_LIB}> expands to ${PYTHON_DEBUG_LIB} when the Debug
  #    configuration is being generated, or nothing (literally) in all other cases.
  # 3. $<$<NOT:$<CONFIG:Debug>>:${PYTHON_RELEASE_LIB}> expands to ${PYTHON_RELEASE_LIB} when
  #    any configuration other than Debug is being generated, or nothing in all other cases.
  # 4. The conditionals in 2 & 3 are mutually exclusive.
  # 5. A logical expression with a conditional that evaluates to 0 yields no value at all.
  #
  # Due to 4 & 5 it's possible to concatenate 2 & 3 to obtain a single value specific to each
  # build configuration. In this example the value will be ${PYTHON_DEBUG_LIB} when generating the
  # Debug configuration, or ${PYTHON_RELEASE_LIB} when generating any other configuration.
  # Note that it's imperative that there is no whitespace between the two expressions, otherwise
  # CMake will insert a semicolon between the two.
  set (PYTHON_EXECUTABLE $<$<CONFIG:Debug>:${PYTHON_DEBUG_EXE}>$<$<NOT:$<CONFIG:Debug>>:${PYTHON_RELEASE_EXE}>)
  set (PYTHON_LIBRARY $<$<CONFIG:Debug>:${PYTHON_DEBUG_LIB}>$<$<NOT:$<CONFIG:Debug>>:${PYTHON_RELEASE_LIB}>)
  set (PYTHON_DLL $<$<CONFIG:Debug>:${PYTHON_DEBUG_DLL}>$<$<NOT:$<CONFIG:Debug>>:${PYTHON_RELEASE_DLL}>)

  set (PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
  set (PYTHON_LIBRARY ${PYTHON_LIBRARY} PARENT_SCOPE)
  set (PYTHON_DLL ${PYTHON_DLL} PARENT_SCOPE)
  set (PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} PARENT_SCOPE)

  message("-- LLDB Found PythonExecutable: ${PYTHON_RELEASE_EXE} and ${PYTHON_DEBUG_EXE}")
  message("-- LLDB Found PythonLibs: ${PYTHON_RELEASE_LIB} and ${PYTHON_DEBUG_LIB}")
  message("-- LLDB Found PythonDLL: ${PYTHON_RELEASE_DLL} and ${PYTHON_DEBUG_DLL}")
  message("-- LLDB Found PythonIncludeDirs: ${PYTHON_INCLUDE_DIRS}")
endfunction(find_python_libs_windows)

if (NOT LLDB_DISABLE_PYTHON)

  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    find_python_libs_windows()

    if (NOT LLDB_RELOCATABLE_PYTHON)
      file(TO_CMAKE_PATH "${PYTHON_HOME}" LLDB_PYTHON_HOME)
      add_definitions( -DLLDB_PYTHON_HOME="${LLDB_PYTHON_HOME}" )
    endif()
  else()
    find_package(PythonLibs REQUIRED)
  endif()
  
  if (PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  endif()
endif()

if (LLDB_DISABLE_PYTHON)
  unset(PYTHON_INCLUDE_DIRS)
  unset(PYTHON_LIBRARY)
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

# Disable MSVC warnings
if( MSVC )
  add_definitions(
    -wd4018 # Suppress 'warning C4018: '>=' : signed/unsigned mismatch'
    -wd4068 # Suppress 'warning C4068: unknown pragma'
    -wd4150 # Suppress 'warning C4150: deletion of pointer to incomplete type'
    -wd4251 # Suppress 'warning C4251: T must have dll-interface to be used by clients of class U.'
    -wd4521 # Suppress 'warning C4521: 'type' : multiple copy constructors specified'
    -wd4530 # Suppress 'warning C4530: C++ exception handler used, but unwind semantics are not enabled.'
  )
endif()

# Use the Unicode (UTF-16) APIs by default on Win32
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
	add_definitions( /D _UNICODE /D UNICODE )
endif()

set(LLDB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLDB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# If building on a 32-bit system, make sure off_t can store offsets > 2GB
if( CMAKE_SIZEOF_VOID_P EQUAL 4 )
  add_definitions( -D_LARGEFILE_SOURCE )
  add_definitions( -D_FILE_OFFSET_BITS=64 )
endif()

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

if (NOT LIBXML2_FOUND AND NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
  # Skip Libxml2 on Windows.  In CMake 3.4 and higher, the algorithm for
  # finding libxml2 got "smarter", and it can now locate the version which is
  # in gnuwin32, even though that version does not contain the headers that
  # LLDB uses.
  find_package(LibXml2)
endif()

# Find libraries or frameworks that may be needed
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_library(CARBON_LIBRARY Carbon)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(CORE_FOUNDATION_LIBRARY CoreFoundation)
  find_library(CORE_SERVICES_LIBRARY CoreServices)
  find_library(SECURITY_LIBRARY Security)
  find_library(DEBUG_SYMBOLS_LIBRARY DebugSymbols PATHS "/System/Library/PrivateFrameworks")

  add_definitions( -DLIBXML2_DEFINED )
  list(APPEND system_libs xml2 ${CURSES_LIBRARIES})
  list(APPEND system_libs ${CARBON_LIBRARY} ${FOUNDATION_LIBRARY}
  ${CORE_FOUNDATION_LIBRARY} ${CORE_SERVICES_LIBRARY} ${SECURITY_LIBRARY}
  ${DEBUG_SYMBOLS_LIBRARY})

else()
  if (LIBXML2_FOUND)
    add_definitions( -DLIBXML2_DEFINED )
    list(APPEND system_libs ${LIBXML2_LIBRARIES})
    include_directories(${LIBXML2_INCLUDE_DIR})
  endif()

endif()

if (HAVE_LIBPTHREAD)
  list(APPEND system_libs pthread)
endif(HAVE_LIBPTHREAD)

if (HAVE_LIBDL)
  list(APPEND system_libs ${CMAKE_DL_LIBS})
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Check for syscall used by lldb-server on linux.
    # If these are not found, it will fall back to ptrace (slow) for memory reads.
    check_cxx_source_compiles("
        #include <sys/uio.h>
        int main() { process_vm_readv(0, nullptr, 0, nullptr, 0, 0); return 0; }"
        HAVE_PROCESS_VM_READV)

    if (HAVE_PROCESS_VM_READV)
        add_definitions(-DHAVE_PROCESS_VM_READV)
    else()
        # If we don't have the syscall wrapper function, but we know the syscall number, we can
        # still issue the syscall manually
        check_cxx_source_compiles("
            #include <sys/syscall.h>
            int main() { return __NR_process_vm_readv; }"
            HAVE_NR_PROCESS_VM_READV)

        if (HAVE_NR_PROCESS_VM_READV)
            add_definitions(-DHAVE_NR_PROCESS_VM_READV)
        endif()
    endif()
endif()

# Figure out if lldb could use lldb-server.  If so, then we'll
# ensure we build lldb-server when an lldb target is being built.
if ((CMAKE_SYSTEM_NAME MATCHES "Darwin") OR
    (CMAKE_SYSTEM_NAME MATCHES "FreeBSD") OR
    (CMAKE_SYSTEM_NAME MATCHES "Linux") OR
    (CMAKE_SYSTEM_NAME MATCHES "NetBSD"))
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

if(MSVC)
    set(LLDB_USE_BUILTIN_DEMANGLER ON)
else()
    option(LLDB_USE_BUILTIN_DEMANGLER "Use lldb's builtin demangler instead of the system one" ON)
endif()
if(LLDB_USE_BUILTIN_DEMANGLER)
    add_definitions(-DLLDB_USE_BUILTIN_DEMANGLER)
endif()
