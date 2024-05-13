add_custom_target(bolt-unit-test-deps)
add_dependencies(bolt-unit-test-deps bolt-test-depends)

add_lit_testsuites(BOLT-UNIT
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS bolt-unit-test-deps)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(check-bolt-unit "Running bolt unit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS bolt-unit-test-deps)
