set(SHADOWCALLSTACK_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHADOWCALLSTACK_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SHADOWCALLSTACK_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-shadowcallstack "Running the ShadowCallStack tests"
  ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(check-shadowcallstack PROPERTIES FOLDER "Compiler-RT Misc")
