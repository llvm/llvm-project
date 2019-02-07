# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY "Only include toolchain files in the 'install' target." OFF)

  set(LLDB_PATH_TO_LLVM_BUILD "" CACHE PATH "Path to LLVM build tree")
  set(LLDB_PATH_TO_CLANG_BUILD "${LLDB_PATH_TO_LLVM_BUILD}" CACHE PATH "Path to Clang build tree")

	# BEGIN Swift Mods
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_LLVM_BUILD}" LLDB_PATH_TO_LLVM_BUILD)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_CLANG_BUILD}" LLDB_PATH_TO_CLANG_BUILD)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_SWIFT_BUILD}" LLDB_PATH_TO_SWIFT_BUILD)

  file(TO_CMAKE_PATH "${LLDB_PATH_TO_CLANG_SOURCE}" LLDB_PATH_TO_CLANG_SOURCE)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_SWIFT_SOURCE}" LLDB_PATH_TO_SWIFT_SOURCE)
  # END Swift Mods

  find_package(LLVM REQUIRED CONFIG
    HINTS "${LLDB_PATH_TO_LLVM_BUILD}" NO_CMAKE_FIND_ROOT_PATH)
  #find_package(Clang REQUIRED CONFIG
  #  HINTS "${LLDB_PATH_TO_CLANG_BUILD}" NO_CMAKE_FIND_ROOT_PATH)

  # We set LLVM_CMAKE_PATH so that GetSVN.cmake is found correctly when building SVNVersion.inc
  set(LLVM_CMAKE_PATH ${LLVM_CMAKE_DIR} CACHE PATH "Path to LLVM CMake modules")

  set(LLVM_MAIN_SRC_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
  set(LLVM_MAIN_INCLUDE_DIR ${LLVM_BUILD_MAIN_INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_LIBRARY_DIR ${LLVM_BUILD_LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_BINARY_DIR ${LLVM_BUILD_BINARY_DIR} CACHE PATH "Path to LLVM build tree")
  set(LLVM_DEFAULT_EXTERNAL_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit CACHE PATH "Path to llvm-lit")

  if(CMAKE_CROSSCOMPILING)
    set(LLVM_NATIVE_BUILD "${LLDB_PATH_TO_LLVM_BUILD}/NATIVE")
    if (NOT EXISTS "${LLVM_NATIVE_BUILD}")
      message(FATAL_ERROR
        "Attempting to cross-compile LLDB standalone but no native LLVM build
        found. Please cross-compile LLVM as well.")
    endif()

    if (CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
      set(HOST_EXECUTABLE_SUFFIX ".exe")
    endif()

    if (NOT CMAKE_CONFIGURATION_TYPES)
      set(LLVM_TABLEGEN_EXE
        "${LLVM_NATIVE_BUILD}/bin/llvm-tblgen${HOST_EXECUTABLE_SUFFIX}")
    else()
      # NOTE: LLVM NATIVE build is always built Release, as is specified in
      # CrossCompile.cmake
      set(LLVM_TABLEGEN_EXE
        "${LLVM_NATIVE_BUILD}/Release/bin/llvm-tblgen${HOST_EXECUTABLE_SUFFIX}")
    endif()
  else()
    find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
      NO_DEFAULT_PATH)
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

  # We append the directory in which LLVMConfig.cmake lives. We expect LLVM's
  # CMake modules to be in that directory as well.
  list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
  include(CheckAtomic)

  if (PYTHON_EXECUTABLE STREQUAL "")
    set(Python_ADDITIONAL_VERSIONS 3.5 3.4 3.3 3.2 3.1 3.0 2.7 2.6 2.5)
    include(FindPythonInterp)
    if( NOT PYTHONINTERP_FOUND )
      message(FATAL_ERROR
              "Unable to find Python interpreter, required for builds and testing.
               Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
    endif()
  else()
    message("-- Found PythonInterp: ${PYTHON_EXECUTABLE}")
  endif()

  # Start Swift Mods
  find_package(Clang REQUIRED CONFIG
    HINTS "${LLDB_PATH_TO_CLANG_BUILD}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  find_package(Swift REQUIRED CONFIG
    HINTS "${LLDB_PATH_TO_SWIFT_BUILD}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  # End Swift Mods

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories(
    "${CMAKE_BINARY_DIR}/include"
    "${LLVM_INCLUDE_DIRS}"
    "${CLANG_INCLUDE_DIRS}"
    "${LLDB_PATH_TO_SWIFT_BUILD}/include"
    "${LLDB_PATH_TO_SWIFT_SOURCE}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/source")

  link_directories("${LLVM_LIBRARY_DIR}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(LLDB_BUILT_STANDALONE 1)
else()
  set(LLDB_PATH_TO_SWIFT_BUILD ${CMAKE_BINARY_DIR})
endif()
