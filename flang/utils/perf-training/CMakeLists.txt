include(LLVMExternalProjectUtils)

set(FLANG_PGO_TRAINING_DATA "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH
  "The path to a lit testsuite containing samples for PGO and order file generation"
  )
set(FLANG_PGO_TRAINING_DATA_SOURCE_DIR OFF CACHE STRING "Path to source directory containing cmake project with source files to use for generating flang pgo data")
set(FLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS "" CACHE STRING "Extra CMake flags to pass to the cmake project with source files to use for generating flang pgo data")
set(FLANG_PGO_TRAINING_DEPS "" CACHE STRING "Extra dependencies needed to build the PGO training data.")
set(FLANG_PGO_TRAINING_CLANG_COUPLING ON CACHE BOOL "Train clang and flang together")
if(FLANG_PGO_TRAINING_CLANG_COUPLING)
  set(CLANG_PGO_TRAINING_DATA_SOURCE_DIR OFF CACHE STRING "Path to source directory containing cmake project with source files to use for generating clang pgo data")
  set(CLANG_PGO_TRAINING_DATA_SOURCE_CMAKE_ARGS "" CACHE STRING "Extra CMake flags to pass to the cmake project with source files to use for generating clang pgo data")
endif()

set(CLANG_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../../clang/utils/perf-training")

set(PERF_HELPER "${LLVM_MAIN_SRC_DIR}/../clang/utils/perf-training/perf-helper.py")

add_custom_target(clear-flang-perf-data
  COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}" perf.data
  COMMENT "Clearing old flang perf data")

option(FLANG_PGO_TRAINING_USE_LLVM_BUILD "Use LLVM build for generating PGO data" ON)

llvm_canonicalize_cmake_booleans(
  FLANG_PGO_TRAINING_USE_LLVM_BUILD
)

if(LLVM_BUILD_INSTRUMENTED)
  configure_lit_site_cfg(
    "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in"
    "${CMAKE_CURRENT_BINARY_DIR}/pgo-data/lit.site.cfg"
    )

  add_lit_testsuite(generate-flang-profraw "Generating flang PGO data"
    "${CMAKE_CURRENT_BINARY_DIR}/pgo-data/"
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS flang flang-rt
    )
  if(FLANG_PGO_TRAINING_CLANG_COUPLING)
    add_lit_testsuite(generate-clang-profraw "Generating clang PGO data"
      "${CLANG_CURRENT_BINARY_DIR}/pgo-data/"
      EXCLUDE_FROM_CHECK_ALL
      DEPENDS clang
      )
  endif()

  add_custom_target(clear-flang-profraw
    COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_BINARY_DIR}/profiles/" profraw
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
      "${CMAKE_CURRENT_BINARY_DIR}"
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
    add_custom_target(generate-flang-profdata DEPENDS ${PROFDATA})

    if(FLANG_PGO_TRAINING_CLANG_COUPLING)
      string(TOUPPER "${LLVM_BUILD_INSTRUMENTED}" LLVM_BUILD_UPPER_INSTRUMENTED)
      if (LLVM_BUILD_UPPER_INSTRUMENTED STREQUAL "CSSPGO")
        message(STATUS "CSSPGO of flang is not supported")
      else()
        add_dependencies(clang-bootstrap-deps generate-flang-profdata)
      endif()
    endif()
  endif()
endif()

if(FLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  configure_lit_site_cfg(
    "${CMAKE_CURRENT_SOURCE_DIR}/bolt.lit.site.cfg.in"
    "${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/lit.site.cfg"
    )

  add_lit_testsuite(generate-flang-bolt-fdata "Generating BOLT profile for flang"
    "${CMAKE_CURRENT_BINARY_DIR}/bolt-fdata/"
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS clear-flang-bolt-fdata clear-flang-perf-data
    )

  add_custom_target(clear-flang-bolt-fdata
    COMMAND "${Python3_EXECUTABLE}" "${PERF_HELPER}" clean "${CMAKE_CURRENT_BINARY_DIR}" fdata
    COMMENT "Clearing old flang BOLT fdata")

endif()
