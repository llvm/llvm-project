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

llvm_canonicalize_cmake_booleans(
  CLANG_ENABLE_STATIC_ANALYZER)

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
  # For the clang-apply-replacements test that uses clang-rename.
  clang-rename

  # For the clang-doc tests that emit bitcode files.
  llvm-bcanalyzer

  # Individual tools we test.
  clang-apply-replacements
  clang-change-namespace
  clangd
  clang-doc
  clang-include-fixer
  clang-move
  clang-query
  clang-reorder-fields
  find-all-symbols
  modularize
  pp-trace

  # These individual tools have no tests, add them here to make them compile
  # together with check-clang-tools, so that we won't break them in the future.
  clangd-indexer

  # Unit tests
  ExtraToolsUnitTests
  )

if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND CLANG_TOOLS_TEST_DEPS
    # For the clang-tidy libclang integration test.
    c-index-test
    # clang-tidy tests require it.
    clang-headers

    clang-tidy
    )
endif()

set(llvm_utils
  FileCheck count not
  )

foreach(t ${llvm_utils})
  if(TARGET ${t})
    list(APPEND CLANG_TOOLS_TEST_DEPS ${t})
  endif()
endforeach()

add_lit_testsuite(check-clang-tools "Running the Clang extra tools' regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANG_TOOLS_TEST_DEPS}
  ARGS ${CLANG_TOOLS_TEST_EXTRA_ARGS}
  )

set_target_properties(check-clang-tools PROPERTIES FOLDER "Clang extra tools' tests")
