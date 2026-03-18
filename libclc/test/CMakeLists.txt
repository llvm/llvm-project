set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
string(REPLACE ";" " " LIBCLC_TARGETS_TO_BUILD "${LIBCLC_TARGETS_TO_BUILD}")

set(LIBCLC_TEST_DEPS
  libclc-opencl-builtins
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-libclc
  "Running the libclc regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LIBCLC_TEST_DEPS}
)
