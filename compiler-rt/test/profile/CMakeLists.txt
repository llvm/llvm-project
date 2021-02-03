set(PROFILE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROFILE_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(PROFILE_TESTSUITES)
set(PROFILE_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND PROFILE_TEST_DEPS profile llvm-profdata llvm-cov)
  if(NOT APPLE AND COMPILER_RT_HAS_LLD AND TARGET lld)
    list(APPEND PROFILE_TEST_DEPS lld)
  endif()
endif()

set(PROFILE_TEST_ARCH ${PROFILE_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(PROFILE_SUPPORTED_ARCH PROFILE_TEST_ARCH)
endif()

foreach(arch ${PROFILE_TEST_ARCH})
  set(PROFILE_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} PROFILE_TEST_TARGET_CC PROFILE_TEST_TARGET_CFLAGS)
  set(CONFIG_NAME Profile-${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
   )
  list(APPEND PROFILE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-profile "Running the profile tests"
  ${PROFILE_TESTSUITES}
  DEPENDS ${PROFILE_TEST_DEPS})
set_target_properties(check-profile PROPERTIES FOLDER "Compiler-RT Misc")
