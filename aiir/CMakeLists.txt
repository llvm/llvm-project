# AIIR project.
cmake_minimum_required(VERSION 3.20.0)
set(LLVM_SUBPROJECT_TITLE "AIIR")

if(NOT DEFINED LLVM_COMMON_CMAKE_UTILS)
  set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
endif()
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

# Check if AIIR is built as a standalone project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(aiir)
  set(AIIR_STANDALONE_BUILD TRUE)
endif()

# Must go below project(..)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(AIIR_STANDALONE_BUILD)
  find_package(LLVM CONFIG REQUIRED)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})
  list(APPEND CMAKE_REQUIRED_DEFINITIONS ${LLVM_DEFINITIONS_LIST})

  include(HandleLLVMOptions)
  include(AddLLVM)
  include(TableGen)

  include_directories(${LLVM_INCLUDE_DIRS})

  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

  # These definitions are needed to fill SHLIBDIR in tests.
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
  if(WIN32 OR CYGWIN)
    # DLL platform -- put DLLs into bin.
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  endif()
  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
endif()

set(AIIR_TOOLS_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path for binary subdirectory (defaults to '${CMAKE_INSTALL_BINDIR}')")
mark_as_advanced(AIIR_TOOLS_INSTALL_DIR)

set(AIIR_MAIN_SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}  )
set(AIIR_MAIN_INCLUDE_DIR ${AIIR_MAIN_SRC_DIR}/include )

set(AIIR_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(AIIR_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(AIIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(AIIR_TOOLS_DIR   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(AddAIIR)
include(IRDLToCpp)

# -BSymbolic is incompatible with TypeID
if("${CMAKE_SHARED_LINKER_FLAGS}" MATCHES "-Bsymbolic[^-]")
  message(FATAL_ERROR " AIIR does not support `-Bsymbolic` (see http://llvm.org/pr51420 ),"
          " try `-Bsymbolic-functions` instead.")
endif()

# Forbid implicit function declaration: this may lead to subtle bugs and we
# don't have a reason to support this.
check_c_compiler_flag("-Werror=implicit-function-declaration" C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION)
append_if(C_SUPPORTS_WERROR_IMPLICIT_FUNCTION_DECLARATION "-Werror=implicit-function-declaration" CMAKE_C_FLAGS)

# Warn on undefined macros. This is often an indication that an include to
# `aiir-config.h` or similar is missing.
check_c_compiler_flag("-Wundef" C_SUPPORTS_WUNDEF)
append_if(C_SUPPORTS_WUNDEF "-Wundef" CMAKE_C_FLAGS)
append_if(C_SUPPORTS_WUNDEF "-Wundef" CMAKE_CXX_FLAGS)

# Forbid mismatch between declaration and definition for class vs struct. This is
# harmless on Unix systems, but it'll be a ticking bomb for MSVC/Windows systems
# where it creeps into the ABI.
check_c_compiler_flag("-Werror=mismatched-tags" C_SUPPORTS_WERROR_MISMATCHED_TAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_C_FLAGS)
append_if(C_SUPPORTS_WERROR_MISMATCHED_TAGS "-Werror=mismatched-tags" CMAKE_CXX_FLAGS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # Silence a false positive GCC -Wunused-but-set-parameter warning in
  # constexpr cases. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827
  # for details
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
    check_cxx_compiler_flag("-Wno-unused-but-set-parameter" CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
    append_if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER "-Wno-unused-but-set-parameter" CMAKE_CXX_FLAGS)
  endif()
  # Silence a false positive GCC -Wdeprecated-copy warning in cases where
  # a copy operator is defined through "using" a base class copy operator.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0")
    check_cxx_compiler_flag("-Wno-deprecated-copy" CXX_SUPPORTS_WNO_DEPRECTAED_COPY)
    append_if(CXX_SUPPORTS_WNO_DEPRECTAED_COPY "-Wno-deprecated-copy" CMAKE_CXX_FLAGS)
  endif()
endif()

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
# aiir-generic-headers are dialect-independent.
add_custom_target(aiir-generic-headers)
set_target_properties(aiir-generic-headers PROPERTIES FOLDER "AIIR/Resources")
# aiir-headers may be dialect-dependent.
add_custom_target(aiir-headers)
set_target_properties(aiir-headers PROPERTIES FOLDER "AIIR/Resources")
add_dependencies(aiir-headers aiir-generic-headers)
add_custom_target(aiir-doc)
set_target_properties(aiir-doc PROPERTIES FOLDER "AIIR/Docs")

# Only enable execution engine if the native target is available.
if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(AIIR_ENABLE_EXECUTION_ENGINE_default 1)
else()
  set(AIIR_ENABLE_EXECUTION_ENGINE_default 0)
endif()
option(AIIR_ENABLE_EXECUTION_ENGINE
       "Enable building the AIIR Execution Engine."
       ${AIIR_ENABLE_EXECUTION_ENGINE_default})

# Build the XeVM conversions and run according tests if the SPIRV backend
# is available.
if ("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(AIIR_ENABLE_XEVM_CONVERSIONS 1)
else()
  set(AIIR_ENABLE_XEVM_CONVERSIONS 0)
endif()

set(AIIR_ENABLE_CUDA_RUNNER 0 CACHE BOOL "Enable building the AIIR CUDA runner")
set(AIIR_ENABLE_ROCM_RUNNER 0 CACHE BOOL "Enable building the AIIR ROCm runner")
set(AIIR_ENABLE_SYCL_RUNNER 0 CACHE BOOL "Enable building the AIIR SYCL runner")
set(AIIR_ENABLE_LEVELZERO_RUNNER 0 CACHE BOOL "Enable building the AIIR LevelZero runner")
set(AIIR_ENABLE_SPIRV_CPU_RUNNER 0 CACHE BOOL "Enable building the AIIR SPIR-V cpu runner")
set(AIIR_ENABLE_VULKAN_RUNNER 0 CACHE BOOL "Enable building the AIIR Vulkan runner")
set(AIIR_ENABLE_NVPTXCOMPILER 0 CACHE BOOL
    "Statically link the nvptxlibrary instead of calling ptxas as a subprocess \
    for compiling PTX to cubin")

set(AIIR_ENABLE_PDL_IN_PATTERNMATCH 1 CACHE BOOL "Enable PDL in PatternMatch")

option(AIIR_INCLUDE_TESTS
       "Generate build targets for the AIIR unit tests."
       ${LLVM_INCLUDE_TESTS})

option(AIIR_INCLUDE_INTEGRATION_TESTS
       "Generate build targets for the AIIR integration tests.")

set(AIIR_INSTALL_AGGREGATE_OBJECTS 1 CACHE BOOL
    "Installs object files needed for add_aiir_aggregate to work out of \
    tree. Package maintainers can disable this to exclude these assets if \
    not desired. Enabling this will result in object files being written \
    under lib/objects-{CMAKE_BUILD_TYPE}.")

set(AIIR_BUILD_AIIR_C_DYLIB 0 CACHE BOOL "Builds libAIIR-C shared library.")

set(AIIR_LINK_AIIR_DYLIB ${LLVM_LINK_LLVM_DYLIB} CACHE BOOL
    "Link tools against libAIIR.so")

configure_file(
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Config/aiir-config.h.cmake
  ${AIIR_INCLUDE_DIR}/aiir/Config/aiir-config.h)

#-------------------------------------------------------------------------------
# Python Bindings Configuration
# Requires:
#   The pybind11 library can be found (set with -DPYBIND_DIR=...)
#   The python executable is correct (set with -DPython3_EXECUTABLE=...)
# By default, find_package and probing for installed pybind11 is performed.
# Super projects can set AIIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES=ON to
# disable all package setup and control it themselves.
#-------------------------------------------------------------------------------

set(AIIR_BINDINGS_PYTHON_NB_DOMAIN "aiir"
  CACHE STRING "nanobind domain for AIIR python bindings.")
set(AIIR_PYTHON_PACKAGE_PREFIX "aiir"
  CACHE STRING "Specifies that all AIIR packages are co-located under the
  `AIIR_PYTHON_PACKAGE_PREFIX` top level package (the API has been
  embedded in a relocatable way).")
set(AIIR_ENABLE_BINDINGS_PYTHON 0 CACHE BOOL
       "Enables building of Python bindings.")
set(AIIR_ENABLE_PYTHON_STABLE_ABI 0 CACHE BOOL
       "Build Python bindings against the stable ABI (abi3, PEP 384) for \
       cross-version compatibility. Requires Python 3.12+.")
if(AIIR_ENABLE_PYTHON_STABLE_ABI AND CMAKE_VERSION VERSION_LESS "3.26.0")
  message(FATAL_ERROR
    "AIIR_ENABLE_PYTHON_STABLE_ABI requires CMake >= 3.26 "
    "(Python::SABIModule not available in ${CMAKE_VERSION}).")
endif()
set(AIIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/aiir_core/aiir" CACHE STRING
       "Prefix under install directory to place python bindings")
set(AIIR_DETECT_PYTHON_ENV_PRIME_SEARCH 1 CACHE BOOL
       "Prime the python detection by searching for a full 'Development' \
       component first (temporary while diagnosing environment specific Python \
       detection issues)")
set(AIIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES 0 CACHE BOOL
       "Performs python dev package configuration sufficient to use all AIIR \
       python features. Super-projects that wish to control their own setup \
       must perform an appropriate find_package of Python3 with \
       'Development.Module' and ensure that find_package(pybind11) is \
       satisfied (and keep up to date as requirements evolve).")

set(_aiir_python_stubgen_enabled ON)
# Stubgen doesn't work when cross-compiling (stubgen will run in the host interpreter and then fail
# to find the extension module for the host arch).
# Note: Stubgen requires some extra handling to work properly when sanitizers are enabled,
# so we skip running it in that case now.
if(CMAKE_CROSSCOMPILING OR (NOT LLVM_USE_SANITIZER STREQUAL ""))
  set(_aiir_python_stubgen_enabled OFF)
endif()

option(AIIR_PYTHON_STUBGEN_ENABLED
       "Generate Python type stubs for the AIIR Python bindings."
       ${_aiir_python_stubgen_enabled})

if(AIIR_ENABLE_BINDINGS_PYTHON)
  include(AIIRDetectPythonEnv)
  # Note that both upstream and downstreams often call this macro. It gates
  # internally on the AIIR_CONFIGURE_PYTHON_DEV_PACKAGES option.
  aiir_configure_python_dev_packages()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(BEFORE
  "include"
  ${AIIR_INCLUDE_DIR}
  )

# Adding tools/aiir-tblgen here as calling add_tablegen sets some variables like
# AIIR_TABLEGEN_EXE in PARENT_SCOPE which gets lost if that folder is included
# from another directory like tools
add_subdirectory(tools/aiir-irdl-to-cpp)
add_subdirectory(tools/aiir-linalg-ods-gen)
add_subdirectory(tools/aiir-pdll)
add_subdirectory(tools/aiir-tblgen)
add_subdirectory(tools/aiir-src-sharder)
set(AIIR_TABLEGEN_EXE "${AIIR_TABLEGEN_EXE}" CACHE INTERNAL "")
set(AIIR_TABLEGEN_TARGET "${AIIR_TABLEGEN_TARGET}" CACHE INTERNAL "")
set(AIIR_PDLL_TABLEGEN_EXE "${AIIR_PDLL_TABLEGEN_EXE}" CACHE INTERNAL "")
set(AIIR_PDLL_TABLEGEN_TARGET "${AIIR_PDLL_TABLEGEN_TARGET}" CACHE INTERNAL "")
set(AIIR_SRC_SHARDER_TABLEGEN_EXE "${AIIR_SRC_SHARDER_TABLEGEN_EXE}" CACHE INTERNAL "")
set(AIIR_SRC_SHARDER_TABLEGEN_TARGET "${AIIR_SRC_SHARDER_TABLEGEN_TARGET}" CACHE INTERNAL "")

add_subdirectory(include/aiir)
add_subdirectory(lib)
# C API needs all dialects for registration, but should be built before tests.
add_subdirectory(lib/CAPI)

if (AIIR_INCLUDE_TESTS)
  add_definitions(-DAIIR_INCLUDE_TESTS)
  add_custom_target(AIIRUnitTests)
  set_target_properties(AIIRUnitTests PROPERTIES FOLDER "AIIR/Tests")
  if (TARGET llvm_gtest)
    add_subdirectory(unittests)
  else()
    message(WARNING "gtest not found, unittests will not be available")
  endif()
  add_subdirectory(test)
endif()
# Tools needs to come late to ensure that AIIR_ALL_LIBS is populated.
# Generally things after this point may depend on AIIR_ALL_LIBS or libAIIR.so.
add_subdirectory(tools)

if(AIIR_ENABLE_BINDINGS_PYTHON)
  # Python sources: built extensions come in via lib/Bindings/Python
  add_subdirectory(python)
endif()

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

option(AIIR_INCLUDE_DOCS "Generate build targets for the AIIR docs."
  ${LLVM_INCLUDE_DOCS})
if (AIIR_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/aiir include/aiir-c
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT aiir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "*.pdll"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${AIIR_INCLUDE_DIR}/aiir ${AIIR_INCLUDE_DIR}/aiir-c
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT aiir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-aiir-headers
                             DEPENDS aiir-headers
                             COMPONENT aiir-headers)
  endif()
endif()

# Custom target to install all aiir libraries
add_custom_target(aiir-libraries)
set_target_properties(aiir-libraries PROPERTIES FOLDER "AIIR/Metatargets")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-aiir-libraries
                           DEPENDS aiir-libraries
                           COMPONENT aiir-libraries)
endif()

get_property(AIIR_LIBS GLOBAL PROPERTY AIIR_ALL_LIBS)
if(AIIR_LIBS)
  list(REMOVE_DUPLICATES AIIR_LIBS)
  foreach(lib ${AIIR_LIBS})
    add_dependencies(aiir-libraries ${lib})
    if(NOT LLVM_ENABLE_IDE)
      add_dependencies(install-aiir-libraries install-${lib})
      add_dependencies(install-aiir-libraries-stripped install-${lib}-stripped)
    endif()
  endforeach()
endif()

add_subdirectory(cmake/modules)

if (AIIR_ENABLE_PYTHON_BENCHMARKS)
  add_subdirectory(utils/mbr)
endif()

if(AIIR_STANDALONE_BUILD)
  llvm_distribution_add_targets()
endif()
