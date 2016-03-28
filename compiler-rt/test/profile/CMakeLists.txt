set(PROFILE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROFILE_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(PROFILE_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND PROFILE_TEST_DEPS profile llvm-profdata)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )
add_lit_testsuite(check-profile "Running the profile tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PROFILE_TEST_DEPS})
set_target_properties(check-profile PROPERTIES FOLDER "Profile tests")
