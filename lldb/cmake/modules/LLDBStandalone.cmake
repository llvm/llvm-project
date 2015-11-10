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
      set(LLVM_MAIN_INCLUDE_DIR "${LLVM_MAIN_SRC_DIR}/include")
      list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
    endif()
  endif()

  if (LLDB_PATH_TO_CLANG_SOURCE)
      get_filename_component(CLANG_MAIN_SRC_DIR ${LLDB_PATH_TO_CLANG_SOURCE}
                             ABSOLUTE)
      set(CLANG_MAIN_INCLUDE_DIR "${CLANG_MAIN_SRC_DIR}/include")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake")

  if (LLDB_PATH_TO_LLVM_BUILD)
    get_filename_component(PATH_TO_LLVM_BUILD ${LLDB_PATH_TO_LLVM_BUILD}
                           ABSOLUTE)
  else()
    message(FATAL_ERROR "Please set LLDB_PATH_TO_LLVM_BUILD to the root "
            "directory of LLVM build or install site.")
  endif()

  if (LLDB_PATH_TO_CLANG_BUILD)
    get_filename_component(PATH_TO_CLANG_BUILD ${LLDB_PATH_TO_CLANG_BUILD}
                           ABSOLUTE)
  else()
    message(FATAL_ERROR "Please set LLDB_PATH_TO_CLANG_BUILD to the root "
            "directory of Clang build or install site.")
  endif()


  # These variables are used by add_llvm_library.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(LLVM_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})

  include(AddLLVM)
  include(HandleLLVMOptions)

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
  # Import CMake library targets from LLVM and Clang.
  include("${LLDB_PATH_TO_LLVM_BUILD}/share/llvm/cmake/LLVMConfig.cmake")
  if (EXISTS "${LLDB_PATH_TO_CLANG_BUILD}/share/clang/cmake/ClangConfig.cmake")
      include("${LLDB_PATH_TO_CLANG_BUILD}/share/clang/cmake/ClangConfig.cmake")
  endif()

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

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
