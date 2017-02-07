# Test runner infrastructure for Clang-based tools. This configures the Clang
# test trees for use by Lit, and delegates to LLVM's lit test handlers.
#
# Note that currently we don't support stand-alone builds of Clang, you must
# be building Clang from within a combined LLVM+Clang checkout..

set(CLANG_TOOLS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_TOOLS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

option(CLANG_TOOLS_TEST_USE_VG "Run Clang tools' tests under Valgrind" OFF)
if(CLANG_TOOLS_TEST_USE_VG)
  set(CLANG_TOOLS_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif()

set(CLANG_TOOLS_TEST_DEPS
  # Base line deps.
  FileCheck count not

  # clang-tidy tests require it.
  clang-headers

  # For the clang-tidy libclang integration test.
  c-index-test

  # Individual tools we test.
  clang-apply-replacements
  clang-change-namespace
  clangd
  clang-include-fixer
  clang-move
  clang-query
  clang-rename
  clang-reorder-fields
  clang-tidy
  find-all-symbols
  modularize
  pp-trace

  # Unit tests
  ExtraToolsUnitTests
  )

add_lit_testsuite(check-clang-tools "Running the Clang extra tools' regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANG_TOOLS_TEST_DEPS}
  ARGS ${CLANG_TOOLS_TEST_EXTRA_ARGS}
  )
set_target_properties(check-clang-tools PROPERTIES FOLDER "Clang extra tools' tests")

