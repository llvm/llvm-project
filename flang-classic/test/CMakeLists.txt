#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} FLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

#configure_lit_site_cfg(
#  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
#  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
#  )

option(FLANG_TEST_USE_VG "Run Flang tests under Valgrind" OFF)
if(FLANG_TEST_USE_VG)
  set(FLANG_TEST_EXTRA_ARGS ${FLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

if( NOT FLANG_BUILT_STANDALONE )
  list(APPEND FLANG_TEST_DEPS
    llvm-config
    FileCheck count not
    llc
    llvm-bcanalyzer
    llvm-nm
    llvm-objdump
    llvm-profdata
    llvm-readobj
    llvm-symbolizer
    opt
    )

  if(TARGET llvm-lto)
    list(APPEND FLANG_TEST_DEPS llvm-lto)
  endif()
endif()

add_custom_target(flang-test-depends DEPENDS ${FLANG_TEST_DEPS})
set_target_properties(flang-test-depends PROPERTIES FOLDER "Flang tests")

add_lit_testsuite(check-flang "Running Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPS}
  ARGS ${FLANG_TEST_EXTRA_ARGS}
  )
set_target_properties(check-flang PROPERTIES FOLDER "Flang tests")

add_lit_testsuite(check-flang-long "Running time-consuming Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  PARAMS "${FLANG_TEST_PARAMS};run_long_tests=true"
  DEPENDS ${FLANG_TEST_DEPS}
  ARGS ${FLANG_TEST_EXTRA_ARGS}
  )
set_target_properties(check-flang-long PROPERTIES FOLDER "Flang tests")

add_lit_testsuites(FLANG ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPS}
)

# Add a legacy target spelling: flang-test
add_custom_target(flang-test)
add_dependencies(flang-test check-flang)
set_target_properties(flang-test PROPERTIES FOLDER "Flang tests")
