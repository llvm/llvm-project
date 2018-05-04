cmake_minimum_required(VERSION 3.4.3)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

include(LLDBStandalone)
include(LLDBConfig)
include(AddLLDB)

# Define the LLDB_CONFIGURATION_xxx matching the build type
if( uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
  add_definitions( -DLLDB_CONFIGURATION_DEBUG )
else()
  add_definitions( -DLLDB_CONFIGURATION_RELEASE )
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Windows|Android")
  set(LLDB_DEFAULT_DISABLE_LIBEDIT 1)
else()
  set(LLDB_DEFAULT_DISABLE_LIBEDIT 0)
endif ()

# We need libedit support to go down both the source and
# the scripts directories.
set(LLDB_DISABLE_LIBEDIT ${LLDB_DEFAULT_DISABLE_LIBEDIT} CACHE BOOL "Disables the use of editline.")
if (LLDB_DISABLE_LIBEDIT)
  add_definitions( -DLLDB_DISABLE_LIBEDIT )
endif()

if(APPLE)
  add_definitions(-DLLDB_USE_OS_LOG)
endif()

# add_subdirectory(include)
add_subdirectory(docs)
if (NOT LLDB_DISABLE_PYTHON)
  if(LLDB_USE_SYSTEM_SIX)
    set(SIX_EXTRA_ARGS "--useSystemSix")
  endif()

  set(LLDB_PYTHON_TARGET_DIR ${LLDB_BINARY_DIR}/scripts)
  set(LLDB_WRAP_PYTHON ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp)
  if(LLDB_BUILD_FRAMEWORK)
    set(LLDB_PYTHON_TARGET_DIR
      ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LLDB_FRAMEWORK_INSTALL_DIR})
    set(LLDB_WRAP_PYTHON ${LLDB_PYTHON_TARGET_DIR}/LLDBWrapPython.cpp)
  else()
    # Don't set -m when building the framework.
    set(FINISH_EXTRA_ARGS "-m")
  endif()


  add_subdirectory(scripts)
endif ()

add_subdirectory(source)
add_subdirectory(tools)

option(LLDB_INCLUDE_TESTS "Generate build targets for the LLDB unit tests."
  ${LLVM_INCLUDE_TESTS})
option(LLDB_TEST_USE_CUSTOM_C_COMPILER "Use the C compiler provided via LLDB_TEST_C_COMPILER for building test inferiors (instead of the just-built compiler). Defaults to OFF." OFF)
option(LLDB_TEST_USE_CUSTOM_CXX_COMPILER "Use the C++ compiler provided via LLDB_TEST_CXX_COMPILER for building test inferiors (instead of the just-built compiler). Defaults to OFF." OFF)
if(LLDB_INCLUDE_TESTS)
  if (NOT LLDB_TEST_USE_CUSTOM_C_COMPILER AND TARGET clang)
    set(LLDB_DEFAULT_TEST_C_COMPILER "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    set(LLDB_DEFAULT_TEST_C_COMPILER "")
  endif()

  if (NOT LLDB_TEST_USE_CUSTOM_CXX_COMPILER AND TARGET clang)
    set(LLDB_DEFAULT_TEST_CXX_COMPILER "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    set(LLDB_DEFAULT_TEST_CXX_COMPILER "")
  endif()

  set(LLDB_TEST_C_COMPILER "${LLDB_DEFAULT_TEST_C_COMPILER}" CACHE PATH "C Compiler to use for building LLDB test inferiors")
  set(LLDB_TEST_CXX_COMPILER "${LLDB_DEFAULT_TEST_CXX_COMPILER}" CACHE PATH "C++ Compiler to use for building LLDB test inferiors")

  if (("${LLDB_TEST_C_COMPILER}" STREQUAL "") OR
      ("${LLDB_TEST_CXX_COMPILER}" STREQUAL ""))
    message(FATAL_ERROR "LLDB test compilers not specified.  Tests will not run")
  endif()

  set(LLDB_TEST_DEPS lldb)

  # darwin-debug is an hard dependency for the testsuite.
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    list(APPEND LLDB_TEST_DEPS darwin-debug)
  endif()

  if(TARGET lldb-server)
    list(APPEND LLDB_TEST_DEPS lldb-server)
  endif()

  if(TARGET debugserver)
    if(NOT CMAKE_HOST_APPLE OR LLDB_CODESIGN_IDENTITY)
      list(APPEND LLDB_TEST_DEPS debugserver)
    endif()
  endif()

  if(TARGET lldb-mi)
    list(APPEND LLDB_TEST_DEPS lldb-mi)
  endif()

  if(NOT LLDB_BUILT_STANDALONE)
    list(APPEND LLDB_TEST_DEPS yaml2obj dsymutil)
  endif()

  if(TARGET liblldb)
    list(APPEND LLDB_TEST_DEPS liblldb)
  endif()

  if(TARGET clang)
    list(APPEND LLDB_TEST_DEPS clang)
  endif()

  add_subdirectory(test)
  add_subdirectory(unittests)
  add_subdirectory(lit)
  add_subdirectory(utils/lldb-dotest)
endif()

if (NOT LLDB_DISABLE_PYTHON)
    # Add a Post-Build Event to copy over Python files and create the symlink
    # to liblldb.so for the Python API(hardlink on Windows)
    add_custom_target(finish_swig ALL
        COMMAND
           ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py
               --srcRoot=${LLDB_SOURCE_DIR}
               --targetDir=${LLDB_PYTHON_TARGET_DIR}
               --cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}/scripts
               --prefix=${CMAKE_BINARY_DIR}
               --cmakeBuildConfiguration=${CMAKE_CFG_INTDIR}
               --lldbLibDir=lib${LLVM_LIBDIR_SUFFIX}
               ${SIX_EXTRA_ARGS}
               ${FINISH_EXTRA_ARGS}
        VERBATIM
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scripts/lldb.py
        COMMENT "Python script sym-linking LLDB Python API")
    # We depend on liblldb being built before we can do this step.
    add_dependencies(finish_swig liblldb lldb-argdumper)

    # If we build the readline module, we depend on that happening
    # first.
    if (TARGET readline)
        add_dependencies(finish_swig readline)
    endif()

    # Ensure we do the python post-build step when building lldb.
    add_dependencies(lldb finish_swig)

    if(LLDB_BUILD_FRAMEWORK)
      # The target to install libLLDB needs to depend on finish swig so that the
      # framework build properly copies over the Python files.
      add_dependencies(install-liblldb finish_swig)
    endif()

    # Add a Post-Build Event to copy the custom Python DLL to the lldb binaries dir so that Windows can find it when launching
    # lldb.exe or any other executables that were linked with liblldb.
    if (WIN32 AND NOT "${PYTHON_DLL}" STREQUAL "")
        # When using the Visual Studio CMake generator the lldb binaries end up in Release/bin, Debug/bin etc.
        file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin" LLDB_BIN_DIR)
        file(TO_NATIVE_PATH "${PYTHON_DLL}" PYTHON_DLL_NATIVE_PATH)
        add_custom_command(
            TARGET finish_swig
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_DLL_NATIVE_PATH} ${LLDB_BIN_DIR} VERBATIM
            COMMENT "Copying Python DLL to LLDB binaries directory.")
    endif ()
endif ()
