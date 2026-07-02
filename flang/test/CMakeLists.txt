set(FLANG_PGO_TRAINING_DATA_SOURCE_DIR OFF CACHE STRING "Path to source directory containing cmake project with source files to use for generating flang pgo data")
set(FLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS "" CACHE STRING "Extra CMake flags to pass to the cmake project with source files to use for generating flang pgo data")
set(FLANG_PGO_TRAINING_DEPS "" CACHE STRING "Extra dependencies needed to build the PGO training data.")
set(FLANG_PGO_TRAINING_CLANG_COUPLING ON CACHE BOOL "Train clang and flang together")
if(FLANG_PGO_TRAINING_CLANG_COUPLING)
  set(CLANG_PGO_TRAINING_DATA_SOURCE_DIR OFF CACHE STRING "Path to source directory containing cmake project with source files to use for generating clang pgo data")
  set(CLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS "" CACHE STRING "Extra CMake flags to pass to the cmake project with source files to use for generating clang pgo data")
endif()

set(PERF_HELPER "${LLVM_MAIN_SRC_DIR}/../clang/utils/perf-training/perf-helper.py")

add_custom_target(clear-flang-perf-data
  COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}" perf.data
  COMMENT "Clearing old flang perf data")

option(FLANG_PGO_TRAINING_USE_LLVM_BUILD "Use LLVM build for generating PGO data" ON)

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
  FLANG_PGO_TRAINING_USE_LLVM_BUILD
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_ENABLE_PLUGINS
  LLVM_INCLUDE_EXAMPLES
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

set(LIT_LOAD_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test.lit.cfg")
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

if (LLVM_INCLUDE_EXAMPLES AND LLVM_ENABLE_PLUGINS AND NOT (WIN32 OR CYGWIN) AND NOT FLANG_STANDALONE_BUILD)
  list(APPEND FLANG_TEST_DEPENDS Bye)
endif()

if (FLANG_INCLUDE_TESTS)
  if (FLANG_GTEST_AVAIL)
    list(APPEND FLANG_TEST_DEPENDS FlangUnitTests)
  endif()
endif()

if (LLVM_INCLUDE_EXAMPLES)
  list(APPEND FLANG_TEST_DEPENDS
    flangPrintFunctionNames
    flangOmpReport
    flangFeatureList
    )
endif ()

if ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES AND NOT FLANG_STANDALONE_BUILD)
  # Flang tests need runtimes module files (flang-rt-mod, libomp-mod, etc.).
  add_custom_target(flang-rt-test-deps)
  list(APPEND FLANG_TEST_DEPENDS flang-rt-test-deps)
endif ()

set(FLANG_BOLT_ALLOWLIST INSTRUMENT PERF LBR)
set(FLANG_BOLT OFF CACHE STRING "Apply BOLT optimization to flang. \
May be specified as one of ${FLANG_BOLT_ALLOWLIST} to use a particular profiling \
  mechanism.")
set_property(CACHE FLANG_BOLT PROPERTY STRINGS OFF ${FLANG_BOLT_ALLOWLIST})
string(TOUPPER "${FLANG_BOLT}" FLANG_BOLT)
if (FLANG_BOLT AND NOT FLANG_BOLT IN_LIST FLANG_BOLT_ALLOWLIST)
    message(FATAL_ERROR "Specified FLANG_BOLT value '${FLANG_BOLT}' is not one of ${FLANG_BOLT_ALLOWLIST}.")
endif()

if (FLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  list(APPEND FLANG_TEST_DEPENDS
    clear-flang-bolt-fdata
    llvm-bolt
    clang-bolt)
endif()

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

include(LLVMExternalProjectUtils)

if(LLVM_BUILD_INSTRUMENTED)
  set(CLANG_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../clang/utils/perf-training")
  set(LIT_LOAD_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/pgo.lit.cfg")
  configure_lit_site_cfg(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pgo-data/lit.site.cfg"
    MAIN_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
    )

  add_lit_testsuite(generate-flang-profraw "Generating flang PGO data"
    "${CMAKE_CURRENT_BINARY_DIR}/pgo-data/"
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS ${FLANG_TEST_DEPENDS}
    )
  if(FLANG_PGO_TRAINING_CLANG_COUPLING)
    add_lit_testsuite(generate-clang-profraw "Generating clang PGO data"
      "${CLANG_CURRENT_BINARY_DIR}/pgo-data/"
      EXCLUDE_FROM_CHECK_ALL
      DEPENDS clang
      )
  endif()

  add_custom_target(clear-flang-profraw
    COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}/.." "${CMAKE_BINARY_DIR}/profiles/" profraw
    COMMENT "Clearing old flang profraw data")
  if (FLANG_PGO_TRAINING_CLANG_COUPLING)
    add_dependencies(clear-flang-profraw clear-profraw)
  endif()

  if(NOT LLVM_PROFDATA)
    find_program(LLVM_PROFDATA llvm-profdata)
  endif()

  if(NOT LLVM_PROFDATA)
    message(STATUS "To enable merging PGO data LLVM_PROFDATA has to point to llvm-profdata")
  else()
    set(PROFDATA "${CMAKE_BINARY_DIR}/flang.profdata")
    set(PROFRAW_TARGETS "")
    set(PROFRAW_DIRS "")
    set(PROFRAW_DEPS "")
    if(FLANG_PGO_TRAINING_CLANG_COUPLING)
      list(APPEND PROFRAW_TARGETS generate-clang-profraw)
      list(APPEND PROFRAW_DIRS "${CLANG_CURRENT_BINARY_DIR}")
      list(APPEND PROFRAW_DEPS clang)
      if (CLANG_PGO_TRAINING_DATA_SOURCE_DIR)
        llvm_ExternalProject_Add(generate-clang-profraw-external
                                 "${CLANG_PGO_TRAINING_DATA_SOURCE_DIR}"
                                 USE_TOOLCHAIN
                                 EXCLUDE_FROM_ALL
                                 NO_INSTALL
                                 CMAKE_ARGS "${CLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS}")
        list(APPEND PROFRAW_TARGETS generate-clang-profraw-external)
      endif()
    endif()
    list(APPEND PROFRAW_TARGETS generate-flang-profraw)
    list(APPEND PROFRAW_DIRS
      "${CMAKE_CURRENT_BINARY_DIR}/.."
      "${CMAKE_BINARY_DIR}/profiles/")
    list(APPEND PROFRAW_DEPS
      flang
      flang-rt)
    if (FLANG_PGO_TRAINING_DATA_SOURCE_DIR)
      llvm_ExternalProject_Add(generate-flang-profraw-external
                               "${FLANG_PGO_TRAINING_DATA_SOURCE_DIR}"
                               USE_TOOLCHAIN
                               ENABLE_FORTRAN
                               EXCLUDE_FROM_ALL
                               NO_INSTALL
                               CMAKE_ARGS "${FLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS}")
      list(APPEND PROFRAW_TARGETS generate-flang-profraw-external)
    endif()
    add_custom_command(
      OUTPUT "${PROFDATA}"
      # PROFRAW_TARGETS are custom targets which are always considered stale.
      # If we add them here to 'DEPENDS', then it will always execute and running
      # ninja install && ninja check-all will result in the profile data being
      # generated twice, and cause the ninja check-all build to fail with errors like:
      # `ld.lld: error: Function Import: link error: linking module flags 'ProfileSummary': IDs have conflicting values in`
      # Therefore we call these targets manually as part of this custom command,
      # which will only run if flang or ${FLANG_PGO_TRAINING_DEPS} are updated.
      COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target ${PROFRAW_TARGETS}
      COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" merge "${LLVM_PROFDATA}" "${PROFDATA}" ${PROFRAW_DIRS}
      COMMENT "Merging flang profdata"
      DEPENDS ${PROFRAW_DEPS} ${FLANG_PGO_TRAINING_DEPS} clear-flang-profraw
    )
    add_custom_target(generate-flang-profdata DEPENDS "${PROFDATA}")

    if(FLANG_PGO_TRAINING_CLANG_COUPLING)
      string(TOUPPER "${LLVM_BUILD_INSTRUMENTED}" LLVM_BUILD_UPPER_INSTRUMENTED)
      if (LLVM_BUILD_UPPER_INSTRUMENTED STREQUAL "CSSPGO")
        message(STATUS "CSSPGO of flang is not supported")
      else()
        add_dependencies(clang-bootstrap-deps generate-flang-profdata)
        # When clang drives the bootstrap PGO, clang's generate-profdata is
        # also a bootstrap provider, so both pipelines run concurrently and
        # clean/regenerate the same shared profraw dirs, racing each other's
        # merge. Order flang after clang to remove the race. generate-profdata
        # already runs in this case, so this adds no extra work; when flang
        # drives the bootstrap (provider is generate-flang-profdata) the guard
        # keeps it a no-op.
        if(TARGET generate-profdata AND
           "${PGO_OPT_PROFDATA_PROVIDER}" STREQUAL "generate-profdata")
          add_dependencies(generate-flang-profdata generate-profdata)
        endif()
      endif()
    endif()
  endif()
endif()

if (FLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  # Add a flang-bolt target for backwards compatibility.
  add_custom_target(flang-bolt DEPENDS flang-test-depends)

  set(FLANG_BOLT_INSTRUMENTED "flang-bolt.inst" CACHE STRING
    "Name of BOLT-instrumented flang binary")
  set(FLANG_INSTRUMENTED "${LLVM_RUNTIME_OUTPUT_INTDIR}/${FLANG_BOLT_INSTRUMENTED}")
  set(PERF_TRAINING_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(FLANG_BOLT_FDATA "${PERF_TRAINING_BINARY_DIR}/flang-prof.fdata")
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
  )
  set(LIT_COMMAND "${lit_base_dir}/${lit_file_name}")

  set(FLANG_BOLT_INPUTS "$<TARGET_FILE:flang>")
  set(FLANG_INSTRUMENTED_OUTPUTS "${FLANG_INSTRUMENTED}")

  # Add in dynamically linked libraries, if needs be. Currently only supported
  # on Linux because it relies on LD_PRELOAD for instrumentation.
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (LLVM_LINK_LLVM_DYLIB)
      set(LLVM_BOLT_INSTRUMENTED "LLVM-bolt.inst" CACHE STRING
        "Name of BOLT-instrumented LLVM library")
      set(LLVM_INSTRUMENTED "${LLVM_RUNTIME_OUTPUT_INTDIR}/${LLVM_BOLT_INSTRUMENTED}")
      list(APPEND FLANG_BOLT_INPUTS "$<TARGET_FILE:LLVM>")
      list(APPEND FLANG_INSTRUMENTED_OUTPUTS "${LLVM_INSTRUMENTED}")
    endif()
  endif()

  # This POST_BUILD command is executed unconditionally even if the flang target
  # is already built.  We need to wrap the whole bolt optimization process in
  # a single python wrapper, so that we can first check if the binary has
  # already been optimized and then exit early with a 0 status if it has.
  add_custom_command(
    TARGET flang-test-depends POST_BUILD
    COMMAND  "${Python3_EXECUTABLE}" "${PERF_HELPER}"
             bolt-optimize
             --method "${FLANG_BOLT}"
             --input "${FLANG_BOLT_INPUTS}"
             --instrumented-output "${FLANG_INSTRUMENTED_OUTPUTS}"
             --fdata "${FLANG_BOLT_FDATA}"
             --perf-training-binary-dir "${PERF_TRAINING_BINARY_DIR}"
             --readelf "$<TARGET_FILE:llvm-readobj>"
             --bolt "$<TARGET_FILE:llvm-bolt>"
             --lit "${LIT_COMMAND}"
             --merge-fdata "$<TARGET_FILE:merge-fdata>"
    COMMENT "Optimizing flang with BOLT"
    USES_TERMINAL
    VERBATIM
  )

  set(LIT_LOAD_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/bolt.lit.cfg")
  configure_lit_site_cfg(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/lit.site.cfg"
    MAIN_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
    )

  add_lit_testsuite(generate-flang-bolt-fdata "Generating BOLT profile for flang"
    "${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/"
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS flang-test-depends clear-flang-bolt-fdata clear-flang-perf-data
    )

  add_custom_target(clear-flang-bolt-fdata
    COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}" fdata
    COMMENT "Clearing old flang BOLT fdata")

endif()
