# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY "Only include toolchain files in the 'install' target." OFF)

  set(LLDB_PATH_TO_LLVM_BUILD "" CACHE PATH "Path to LLVM build tree")
  set(LLDB_PATH_TO_CLANG_BUILD "${LLDB_PATH_TO_LLVM_BUILD}" CACHE PATH "Path to Clang build tree")

  file(TO_CMAKE_PATH "${LLDB_PATH_TO_LLVM_BUILD}" LLDB_PATH_TO_LLVM_BUILD)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_CLANG_BUILD}" LLDB_PATH_TO_CLANG_BUILD)

  # BEGIN Swift Mods
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_CMARK_BUILD}" LLDB_PATH_TO_CMARK_BUILD)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_SWIFT_BUILD}" LLDB_PATH_TO_SWIFT_BUILD)

  file(TO_CMAKE_PATH "${LLDB_PATH_TO_CLANG_SOURCE}" LLDB_PATH_TO_CLANG_SOURCE)
  file(TO_CMAKE_PATH "${LLDB_PATH_TO_SWIFT_SOURCE}" LLDB_PATH_TO_SWIFT_SOURCE)
  # END Swift Mods

  find_package(LLVM REQUIRED CONFIG
    HINTS "${LLDB_PATH_TO_LLVM_BUILD}" NO_CMAKE_FIND_ROOT_PATH)
  #find_package(Clang REQUIRED CONFIG
  #  HINTS "${LLDB_PATH_TO_CLANG_BUILD}" NO_CMAKE_FIND_ROOT_PATH)

  # We set LLVM_MAIN_INCLUDE_DIR so that it gets included in TableGen flags.
  #set(LLVM_MAIN_INCLUDE_DIR "${LLVM_BUILD_MAIN_INCLUDE_DIR}" CACHE PATH "Path to LLVM's source include dir")
  # We set LLVM_CMAKE_PATH so that GetSVN.cmake is found correctly when building SVNVersion.inc
  set(LLVM_CMAKE_PATH ${LLVM_CMAKE_DIR} CACHE PATH "Path to LLVM CMake modules")

  # BEGIN Swift Mods
  if (LLDB_PATH_TO_CLANG_SOURCE)
    get_filename_component(CLANG_MAIN_SRC_DIR ${LLDB_PATH_TO_CLANG_SOURCE} ABSOLUTE)
    set(CLANG_MAIN_INCLUDE_DIR "${CLANG_MAIN_SRC_DIR}/include")
  endif()

  if (LLDB_PATH_TO_SWIFT_SOURCE)
      get_filename_component(SWIFT_MAIN_SRC_DIR ${LLDB_PATH_TO_SWIFT_SOURCE}
                             ABSOLUTE)
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake")
  list(APPEND CMAKE_MODULE_PATH "${LLDB_PATH_TO_SWIFT_SOURCE}/cmake/modules")
  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_DIR ${LLVM_OBJ_ROOT}/cmake/modules/CMakeFiles CACHE PATH "Path to LLVM build tree CMake files")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  set(LLVMCONFIG_FILE "${LLDB_PATH_TO_LLVM_BUILD}/lib/cmake/llvm/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()
  # END Swift Mods

  set(LLVM_EXTERNAL_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit CACHE PATH "Path to llvm-lit")
  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  # BEGIN Swift Mods
  get_filename_component(PATH_TO_SWIFT_BUILD ${LLDB_PATH_TO_SWIFT_BUILD}
                         ABSOLUTE)

  get_filename_component(PATH_TO_CMARK_BUILD ${LLDB_PATH_TO_CMARK_BUILD}
                         ABSOLUTE)
  # END Swift Mods

  # These variables are used by add_llvm_library.
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

  # BEGIN Swift Mods
  find_package(Clang REQUIRED CONFIG
    HINTS "${LLDB_PATH_TO_CLANG_BUILD}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  find_package(Swift REQUIRED CONFIG
    HINTS "${PATH_TO_SWIFT_BUILD}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
  # END Swift Mods

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
  set(LLVM_INCLUDE_TESTS ON CACHE INTERNAL "")

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories(
    "${CMAKE_BINARY_DIR}/include"
    "${LLVM_INCLUDE_DIRS}"
    "${CLANG_INCLUDE_DIRS}")

  # BEGIN Swift Mods
  if (EXISTS "${SWIFT_MAIN_SRC_DIR}/include")
    include_directories("${SWIFT_MAIN_SRC_DIR}/include")
  endif()
  # END Swift Mods

  link_directories("${LLVM_LIBRARY_DIR}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(LLDB_BUILT_STANDALONE 1)
else()
  set(LLDB_PATH_TO_SWIFT_BUILD ${CMAKE_BINARY_DIR})
endif()
