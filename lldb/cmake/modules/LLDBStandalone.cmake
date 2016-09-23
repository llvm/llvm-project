# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)
  cmake_minimum_required(VERSION 2.8.12.2)

  if (POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW) # automatic when 2.8.12 is required
  endif()

  option(LLVM_INSTALL_TOOLCHAIN_ONLY "Only include toolchain files in the 'install' target." OFF)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  set(FIND_PATHS "")
  if (LLDB_PATH_TO_LLVM_BUILD)
    set(FIND_PATHS "${LLDB_PATH_TO_LLVM_BUILD}/bin")
  endif()
  find_program(LLVM_CONFIG "llvm-config"
    PATHS ${FIND_PATHS})

  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root")
    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT
    )
    if(NOT HAD_ERROR)
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT})

    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()
  else()
    message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
  endif()

  list(GET CONFIG_OUTPUT 0 ENABLE_ASSERTIONS)
  list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
  list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
  list(GET CONFIG_OUTPUT 3 INCLUDE_DIR)
  list(GET CONFIG_OUTPUT 4 LLVM_OBJ_ROOT)
  list(GET CONFIG_OUTPUT 5 MAIN_SRC_DIR)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

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

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
  endif()


  get_filename_component(PATH_TO_SWIFT_BUILD ${LLDB_PATH_TO_SWIFT_BUILD}
                         ABSOLUTE)

  get_filename_component(PATH_TO_CMARK_BUILD ${LLDB_PATH_TO_CMARK_BUILD}
                         ABSOLUTE)

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

  include(AddLLVM)
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

  # Import CMake library targets from LLVM and Clang.
  if (EXISTS "${LLDB_PATH_TO_CLANG_BUILD}/lib/cmake/clang/ClangConfig.cmake")
      include("${LLDB_PATH_TO_CLANG_BUILD}/lib/cmake/clang/ClangConfig.cmake")
  endif()

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  # Why are we doing this?
  # set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CLANG_MAIN_INCLUDE_DIR "${CLANG_MAIN_SRC_DIR}/include")

  set(SWIFT_MAIN_INCLUDE_DIR "${SWIFT_MAIN_SRC_DIR}/include")

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${LLVM_BINARY_DIR}/include"
                      "${LLVM_BINARY_DIR}/tools/clang/include"
                      "${LLVM_MAIN_INCLUDE_DIR}"
                      "${PATH_TO_CLANG_BUILD}/include"
                      "${CLANG_MAIN_INCLUDE_DIR}"
                      "${PATH_TO_SWIFT_BUILD}/include"
                      "${SWIFT_MAIN_INCLUDE_DIR}"
                      "${CMAKE_CURRENT_SOURCE_DIR}/source")
  link_directories("${LLVM_LIBRARY_DIR}"
                   "${PATH_TO_CLANG_BUILD}/lib${LLVM_LIBDIR_SUFFIX}"
                   "${PATH_TO_SWIFT_BUILD}/lib${LLVM_LIBDIR_SUFFIX}"
                   "${PATH_TO_CMARK_BUILD}/src")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(LLDB_BUILT_STANDALONE 1)
else()
  set(LLDB_PATH_TO_SWIFT_BUILD ${CMAKE_BINARY_DIR})
endif()
