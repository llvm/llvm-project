# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

set(FLANG_INTRINSIC_MODULES_DIR ${FLANG_BINARY_DIR}/tools/f18/include)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_TEST_DEPENDS
  flang
  f18
  llvm-lit
  FileCheck
  count
  not
)
add_lit_testsuite(check-all "Running the Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENS}
  )
set_target_properties(check-all PROPERTIES FOLDER "Tests")

