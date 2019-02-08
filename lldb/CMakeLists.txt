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

if (LLDB_DISABLE_LIBEDIT)
  add_definitions( -DLLDB_DISABLE_LIBEDIT )
else()
  find_package(LibEdit REQUIRED)
endif()

if(APPLE)
  add_definitions(-DLLDB_USE_OS_LOG)
endif()

add_subdirectory(docs)
if (NOT LLDB_DISABLE_PYTHON)
  add_subdirectory(scripts)
endif ()
add_subdirectory(source)
add_subdirectory(tools)

option(LLDB_INCLUDE_TESTS "Generate build targets for the LLDB unit tests." ${LLVM_INCLUDE_TESTS})
option(LLDB_TEST_USE_CUSTOM_C_COMPILER "Use the C compiler provided via LLDB_TEST_C_COMPILER for building test inferiors (instead of the just-built compiler). Defaults to OFF." OFF)
option(LLDB_TEST_USE_CUSTOM_CXX_COMPILER "Use the C++ compiler provided via LLDB_TEST_CXX_COMPILER for building test inferiors (instead of the just-built compiler). Defaults to OFF." OFF)
if(LLDB_INCLUDE_TESTS)

  # Set the path to the default lldb test executable.
  set(LLDB_DEFAULT_TEST_EXECUTABLE "${LLVM_RUNTIME_OUTPUT_INTDIR}/lldb${CMAKE_EXECUTABLE_SUFFIX}")

  # Set the paths to default llvm tools.
  set(LLDB_DEFAULT_TEST_DSYMUTIL "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/dsymutil${CMAKE_EXECUTABLE_SUFFIX}")
  set(LLDB_DEFAULT_TEST_FILECHECK "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/FileCheck${CMAKE_EXECUTABLE_SUFFIX}")

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

  set(LLDB_TEST_EXECUTABLE "${LLDB_DEFAULT_TEST_EXECUTABLE}" CACHE PATH "lldb executable used for testing")
  set(LLDB_TEST_C_COMPILER "${LLDB_DEFAULT_TEST_C_COMPILER}" CACHE PATH "C Compiler to use for building LLDB test inferiors")
  set(LLDB_TEST_CXX_COMPILER "${LLDB_DEFAULT_TEST_CXX_COMPILER}" CACHE PATH "C++ Compiler to use for building LLDB test inferiors")
  set(LLDB_TEST_DSYMUTIL "${LLDB_DEFAULT_TEST_DSYMUTIL}" CACHE PATH "dsymutil used for generating dSYM bundles")
  set(LLDB_TEST_FILECHECK "${LLDB_DEFAULT_TEST_FILECHECK}" CACHE PATH "FileCheck used for testing purposes")

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
    list(APPEND LLDB_TEST_DEPS debugserver)
  endif()

  if(TARGET lldb-mi)
    list(APPEND LLDB_TEST_DEPS lldb-mi)
  endif()

  if(TARGET lldb-instr)
    list(APPEND LLDB_TEST_DEPS lldb-instr)
  endif()

  if(NOT LLDB_BUILT_STANDALONE)
    list(APPEND LLDB_TEST_DEPS yaml2obj)
  endif()

  if(TARGET liblldb)
    list(APPEND LLDB_TEST_DEPS liblldb)
  endif()

  # Add dependencies if we test with the in-tree clang.
  # This works with standalone builds as they import the clang target.
  if(TARGET clang)
    list(APPEND LLDB_TEST_DEPS clang)
    if(APPLE)
      # If we build clang, we should build libcxx.
      # FIXME: Standalone builds should import the cxx target as well.
      if(LLDB_BUILT_STANDALONE)
        # For now check that the include directory exists.
        set(cxx_dir "${LLVM_BINARY_DIR}/include/c++")
        if(NOT EXISTS ${cxx_dir})
          message(WARNING "LLDB test suite requires libc++ in llvm/projects/libcxx or an existing build symlinked to ${cxx_dir}")
        endif()
      else()
        list(APPEND LLDB_TEST_DEPS cxx)
      endif()
    endif()
  endif()

  if(TARGET dsymutil)
    list(APPEND LLDB_TEST_DEPS dsymutil)
  endif()

  add_custom_target(lldb-test-depends DEPENDS ${LLDB_TEST_DEPS})

  add_subdirectory(test)
  add_subdirectory(unittests)
  add_subdirectory(lit)
  add_subdirectory(utils/lldb-dotest)
endif()

if (NOT LLDB_DISABLE_PYTHON)
    if(NOT LLDB_BUILD_FRAMEWORK)
      set(use_python_wrapper_from_src_dir -m)
    endif()
    if(LLDB_USE_SYSTEM_SIX)
      set(use_six_py_from_system --useSystemSix)
    endif()
    get_target_property(lldb_scripts_dir swig_wrapper BINARY_DIR)
    get_target_property(liblldb_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)

    # Add a Post-Build Event to copy over Python files and create the symlink
    # to liblldb.so for the Python API(hardlink on Windows)
    add_custom_target(finish_swig ALL
        COMMAND
           ${PYTHON_EXECUTABLE} ${LLDB_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py
               --srcRoot=${LLDB_SOURCE_DIR}
               --targetDir=${liblldb_build_dir}
               --cfgBldDir=${lldb_scripts_dir}
               --prefix=${CMAKE_BINARY_DIR}
               --cmakeBuildConfiguration=${CMAKE_CFG_INTDIR}
               --lldbLibDir=lib${LLVM_LIBDIR_SUFFIX}
               ${use_python_wrapper_from_src_dir}
               ${use_six_py_from_system}
        VERBATIM
        DEPENDS ${LLDB_SOURCE_DIR}/scripts/finishSwigWrapperClasses.py
        DEPENDS ${lldb_scripts_dir}/lldb.py
        COMMENT "Python script sym-linking LLDB Python API")

    if (TARGET readline)
      set(readline_dep readline)
    endif()
    add_dependencies(finish_swig swig_wrapper liblldb lldb-argdumper ${readline_dep})

    # Ensure we do the python post-build step when building lldb.
    add_dependencies(lldb finish_swig)

    if(LLDB_BUILD_FRAMEWORK)
      add_dependencies(lldb-framework finish_swig)
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
