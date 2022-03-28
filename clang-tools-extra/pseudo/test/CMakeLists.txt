set(CLANG_PSEUDO_TEST_DEPS
  clang-pseudo
  ClangPseudoTests
  )

foreach(dep FileCheck not)
  if(TARGET ${dep})
    list(APPEND CLANG_PSEUDO_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

add_lit_testsuite(check-clang-pseudo "Running the clang-pseudo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANG_PSEUDO_TEST_DEPS})
