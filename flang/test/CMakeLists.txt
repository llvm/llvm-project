# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.
add_subdirectory(lib)

set(FLANG_TEST_Fortran_FLAGS "" CACHE STRING "Additional Fortran flags for running tests, such as -fintrinsic-modules-path=<path>")

if ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
  set(FLANG_TEST_ENABLE_MODULES_default ON)
else ()
  set(FLANG_TEST_ENABLE_MODULES_default OFF)
endif ()
option(FLANG_TEST_ENABLE_MODULES "Force-enable tests that require intrinsic modules from Flang-RT" "${FLANG_TEST_ENABLE_MODULES_default}")


if ("openmp" IN_LIST LLVM_ENABLE_RUNTIMES AND FLANG_TEST_ENABLE_MODULES AND NOT FLANG_STANDALONE_BUILD)
  set(FLANG_TEST_ENABLE_OPENMP_default ON)
else ()
  set(FLANG_TEST_ENABLE_OPENMP_default OFF)
endif ()
option(FLANG_TEST_ENABLE_OPENMP "Force-enable tests that require modules from OpenMP" "${FLANG_TEST_ENABLE_OPENMP_default}")


llvm_canonicalize_cmake_booleans(
  FLANG_STANDALONE_BUILD
  LLVM_BUILD_EXAMPLES
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_ENABLE_PLUGINS
  FLANG_TEST_ENABLE_MODULES
  FLANG_TEST_ENABLE_OPENMP
)

set(FLANG_TOOLS_DIR ${FLANG_BINARY_DIR}/bin)

# Check if 128-bit float computations can be done via long double
check_cxx_source_compiles(
  "#include <cfloat>
   #if LDBL_MANT_DIG != 113
   #error LDBL_MANT_DIG != 113
   #endif
   int main() { return 0; }
  "
  HAVE_LDBL_MANT_DIG_113)

# FIXME In out-of-tree builds, "SHLIBDIR" is undefined and passing it to
# `configure_lit_site_cfg` leads to a configuration error. This is currently
# only required by plugins/examples, which are not supported in out-of-tree
# builds.
if (FLANG_STANDALONE_BUILD)
  set(PATHS_FOR_PLUGINS "")
else ()
  set(PATHS_FOR_PLUGINS "SHLIBDIR")
endif ()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/NonGtestUnit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/NonGtestUnit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/NonGtestUnit/lit.cfg.py
)

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_TEST_DEPENDS
  flang
  fir-opt
  tco
  bbc
  FortranDecimal
)

if (NOT FLANG_STANDALONE_BUILD)
  list(APPEND FLANG_TEST_DEPENDS
    llvm-config
    FileCheck
    count
    not
    llvm-bcanalyzer
    llvm-dis
    llvm-objcopy
    llvm-objdump
    llvm-profdata
    llvm-readelf
    llvm-readobj
    split-file
  )
endif ()

if (LLVM_BUILD_EXAMPLES AND LLVM_ENABLE_PLUGINS AND NOT (WIN32 OR CYGWIN) AND NOT FLANG_STANDALONE_BUILD)
  list(APPEND FLANG_TEST_DEPENDS Bye)
endif()

if (FLANG_INCLUDE_TESTS)
  if (FLANG_GTEST_AVAIL)
    list(APPEND FLANG_TEST_DEPENDS FlangUnitTests)
  endif()
endif()

if (LLVM_BUILD_EXAMPLES)
  list(APPEND FLANG_TEST_DEPENDS
    flangPrintFunctionNames
    flangOmpReport
    flangFeatureList
    )
endif ()

if ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES AND NOT FLANG_STANDALONE_BUILD)
  # For intrinsic module files (in flang-rt/)
  list(APPEND FLANG_TEST_DEPENDS "flang-rt-mod") 
  
  if ("openmp" IN_LIST LLVM_ENABLE_RUNTIMES)
    # For omplib.mod and omplib_kinds.mod (in openmp/)
    list(APPEND FLANG_TEST_DEPENDS "libomp-mod") 
  endif ()
endif ()

add_custom_target(flang-test-depends DEPENDS ${FLANG_TEST_DEPENDS})
set_target_properties(flang-test-depends PROPERTIES FOLDER "Flang/Meta")

add_lit_testsuite(check-flang "Running the Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENDS}
)
set_target_properties(check-flang PROPERTIES FOLDER "Flang/Meta")

# In case of standalone builds.
if (FLANG_STANDALONE_BUILD)
  add_lit_testsuites(FLANG ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${FLANG_TEST_PARAMS}
    DEPENDS ${FLANG_TEST_DEPENDS})
else()
  add_lit_testsuites(FLANG ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS ${FLANG_TEST_PARAMS}
    DEPENDS ${FLANG_TEST_DEPENDS})
endif()

# To modify the default target triple for flang tests.
if (DEFINED FLANG_TEST_TARGET_TRIPLE)
  if (NOT DEFINED LLVM_TARGET_TRIPLE_ENV OR LLVM_TARGET_TRIPLE_ENV STREQUAL "")
    message(FATAL_ERROR "LLVM_TARGET_TRIPLE_ENV must also be defined in order "
                        "to use FLANG_TEST_TARGET_TRIPLE.")
  endif()
endif()
