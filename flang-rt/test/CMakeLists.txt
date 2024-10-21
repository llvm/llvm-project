#===-- test/CMakeLists.txt -------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  FLANG_STANDALONE_BUILD
  LLVM_BUILD_EXAMPLES
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_ENABLE_PLUGINS
)

set(FLANG_TOOLS_DIR ${FLANG_BINARY_DIR}/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

if (TARGET FlangRTUnitTests)
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
endif ()

set(FLANGRT_TEST_DEPENDS
    FlangRTUnitTests
    flang_rt.unittest
  )

add_custom_target(flang-rt-test-depends)
set_target_properties(flang-rt-test-depends PROPERTIES FOLDER "Fortran Runtime/Meta")
add_dependencies(flang-rt-test-depends ${FLANGRT_TEST_DEPENDS})

add_lit_testsuite(check-flang-rt "Running the flang-rt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${FLANGRT_TEST_DEPENDS}
)
set_target_properties(check-flang-rt PROPERTIES FOLDER "Fortran Runtime/Meta")


add_lit_testsuites(flang-rt ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${FLANGRT_TEST_DEPENDS})
