# If we are not building as a part of LLVM, build LLDB as an
# standalone project, using LLVM as an external library:
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(lldb)

  option(LLVM_INSTALL_TOOLCHAIN_ONLY "Only include toolchain files in the 'install' target." OFF)

  # Rely on llvm-config.
  set(CONFIG_OUTPUT)
  find_program(LLVM_CONFIG "llvm-config")
  if(LLVM_CONFIG)
    message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")
    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root"
      "--cmakedir")
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
  list(GET CONFIG_OUTPUT 6 LLVM_CMAKE_PATH)

  if(NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_DIR ${LLVM_OBJ_ROOT}/cmake/modules/CMakeFiles CACHE PATH "Path to LLVM build tree CMake files")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")
  set(LLVM_EXTERNAL_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit CACHE PATH "Path to llvm-lit")

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVMCONFIG_FILE "${LLVM_CMAKE_PATH}/LLVMConfig.cmake")
  if(EXISTS ${LLVMCONFIG_FILE})
    file(TO_CMAKE_PATH "${LLVM_CMAKE_PATH}" LLVM_CMAKE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(${LLVMCONFIG_FILE})
  else()
    message(FATAL_ERROR "Not found: ${LLVMCONFIG_FILE}")
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

  # Import CMake library targets from LLVM and Clang.
  include("${LLVM_OBJ_ROOT}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm/LLVMConfig.cmake")
  # cmake/clang/ClangConfig.cmake is not created when LLVM and Clang are built together.
  if (EXISTS "${LLVM_OBJ_ROOT}/lib${LLVM_LIBDIR_SUFFIX}/cmake/clang/ClangConfig.cmake")
    include("${LLVM_OBJ_ROOT}/lib${LLVM_LIBDIR_SUFFIX}/cmake/clang/ClangConfig.cmake")
  endif()

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR})

  set(CMAKE_INCLUDE_CURRENT_DIR ON)
  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  # Next three include directories are needed when llvm-config is located in build directory.
  # LLVM and Clang are assumed to be built together
  if (EXISTS "${LLVM_OBJ_ROOT}/include")
    include_directories("${LLVM_OBJ_ROOT}/include")
  endif()
  if (EXISTS "${LLVM_MAIN_SRC_DIR}/tools/clang/include")
    include_directories("${LLVM_MAIN_SRC_DIR}/tools/clang/include")
  endif()
  if (EXISTS "${LLVM_OBJ_ROOT}/tools/clang/include")
    include_directories("${LLVM_OBJ_ROOT}/tools/clang/include")
  endif()
  link_directories("${LLVM_LIBRARY_DIR}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

  set(LLDB_BUILT_STANDALONE 1)
endif()
