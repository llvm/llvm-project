set(CLANG_INCLUDE_CLEANER_TEST_DEPS
  ClangIncludeCleanerTests
  )

foreach (dep FileCheck not count)
  if(TARGET ${dep})
    list(APPEND CLANG_INCLUDE_CLEANER_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_BINARY_DIR}/lit.cfg.py)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.cfg.py)

add_lit_testsuite(check-clang-include-cleaner "Running the clang-include-cleaner regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANG_INCLUDE_CLEANER_TEST_DEPS})
