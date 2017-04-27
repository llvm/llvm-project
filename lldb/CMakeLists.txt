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

# add_subdirectory(include)
add_subdirectory(docs)
if (NOT LLDB_DISABLE_PYTHON)
  if(LLDB_USE_SYSTEM_SIX)
    set(SIX_EXTRA_ARGS "--useSystemSix")
  endif()

  set(LLDB_PYTHON_TARGET_DIR ${LLDB_BINARY_DIR}/scripts)
  if(LLDB_BUILD_FRAMEWORK)
    set(LLDB_PYTHON_TARGET_DIR
      ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LLDB_FRAMEWORK_INSTALL_DIR})
  else()
    # Don't set -m when building the framework.
    set(FINISH_EXTRA_ARGS "-m")
  endif()
  set(LLDB_WRAP_PYTHON ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp)

  add_subdirectory(scripts)
endif ()
add_subdirectory(source)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(lit)

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
