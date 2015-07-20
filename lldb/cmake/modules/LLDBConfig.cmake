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

set(LLDB_ENABLE_PYTHON_SCRIPTS_SWIG_API_GENERATION 1 CACHE BOOL
  "Enables using new Python scripts for SWIG API generation .")
set(LLDB_RELOCATABLE_PYTHON 0 CACHE BOOL
  "Causes LLDB to use the PYTHONHOME environment variable to locate Python.")

if ((NOT MSVC) OR MSVC12)
  add_definitions( -DHAVE_ROUND )
endif()

if (LLDB_DISABLE_CURSES)
  add_definitions( -DLLDB_DISABLE_CURSES )
endif()

if (NOT LLDB_DISABLE_PYTHON)
  if(UNIX)
    # This is necessary for crosscompile on Ubuntu 14.04 64bit. Need a proper fix.
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
    endif()
  endif()

  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    if (NOT "${PYTHON_HOME}" STREQUAL "")
      file(TO_CMAKE_PATH "${PYTHON_HOME}" PYTHON_HOME)
      if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        file(TO_CMAKE_PATH "${PYTHON_HOME}/python_d.exe" PYTHON_EXECUTABLE)
        file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/python27_d.lib" PYTHON_LIBRARY)
        file(TO_CMAKE_PATH "${PYTHON_HOME}/python27_d.dll" PYTHON_DLL)
      else()
        file(TO_CMAKE_PATH "${PYTHON_HOME}/python.exe" PYTHON_EXECUTABLE)
        file(TO_CMAKE_PATH "${PYTHON_HOME}/libs/python27.lib" PYTHON_LIBRARY)
        file(TO_CMAKE_PATH "${PYTHON_HOME}/python27.dll" PYTHON_DLL)
      endif()

      file(TO_CMAKE_PATH "${PYTHON_HOME}/Include" PYTHON_INCLUDE_DIR)
      if (NOT LLDB_RELOCATABLE_PYTHON)
        add_definitions( -DLLDB_PYTHON_HOME="${PYTHON_HOME}" )
      endif()
    else()
      message("Embedding Python on Windows without specifying a value for PYTHON_HOME is deprecated.  Support for this will be dropped soon.")

      if ("${PYTHON_INCLUDE_DIR}" STREQUAL "" OR "${PYTHON_LIBRARY}" STREQUAL "")
        message("-- LLDB Embedded python disabled.  Embedding python on Windows requires "
                "manually specifying PYTHON_INCLUDE_DIR *and* PYTHON_LIBRARY")
        set(LLDB_DISABLE_PYTHON 1)
      endif()
    endif()

    if (PYTHON_LIBRARY)
      message("-- Found PythonLibs: ${PYTHON_LIBRARY}")
      include_directories(${PYTHON_INCLUDE_DIR})
    endif()

  else()
    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
  endif()
endif()

if (LLDB_DISABLE_PYTHON)
  unset(PYTHON_INCLUDE_DIR)
  unset(PYTHON_LIBRARY)
  add_definitions( -DLLDB_DISABLE_PYTHON )
endif()

include_directories(../clang/include)
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

# Disable Clang warnings
check_cxx_compiler_flag("-Wno-deprecated-register"
                        CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
if (CXX_SUPPORTS_NO_DEPRECATED_REGISTER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
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

if (CMAKE_VERSION VERSION_LESS 2.8.12)
  set(cmake_2_8_12_INTERFACE)
  set(cmake_2_8_12_PRIVATE)
  set(cmake_2_8_12_PUBLIC)
else ()
  set(cmake_2_8_12_INTERFACE INTERFACE)
  set(cmake_2_8_12_PRIVATE PRIVATE)
  set(cmake_2_8_12_PUBLIC PUBLIC)
endif ()

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

if (NOT LIBXML2_FOUND)
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
  list(APPEND system_libs xml2 ncurses panel)
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
