set(CTX_PROFILE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CTX_PROFILE_TESTSUITES)

# Add unit tests.
if(COMPILER_RT_INCLUDE_TESTS)
  foreach(arch ${CTX_PROFILE_SUPPORTED_ARCH})
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND CTX_PROFILE_TEST_DEPS CtxProfileUnitTests)
    list(APPEND CTX_PROFILE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
  endforeach()
endif()

add_lit_testsuite(check-ctx_profile "Running the Contextual Profiler tests"
  ${CTX_PROFILE_TESTSUITES}
  DEPENDS ${CTX_PROFILE_TEST_DEPS})
set_target_properties(check-ctx_profile PROPERTIES FOLDER "Compiler-RT Misc")
