set(CLANGD_TEST_DEPS
  clangd
  ClangdTests
  # No tests for these, but we should still make sure they build.
  clangd-indexer
  dexp
  )

if(CLANGD_BUILD_XPC)
  list(APPEND CLANGD_TEST_DEPS clangd-xpc-test-client)
endif()

foreach(dep FileCheck count not)
  if(TARGET ${dep})
    list(APPEND CLANGD_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/../unittests/lit.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/../unittests/lit.cfg)

add_lit_testsuite(check-clangd "Running the Clangd regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}/Unit;${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANGD_TEST_DEPS})
