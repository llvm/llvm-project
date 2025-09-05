set(PFP_TESTSUITES)
set(PFP_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} unwind compiler-rt)
if(COMPILER_RT_HAS_LLD AND TARGET lld)
  list(APPEND PFP_TEST_DEPS lld)
endif()

macro(add_pfp_testsuite arch thinlto)
  set(PFP_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} PFP_TEST_TARGET_CC PFP_TEST_TARGET_CFLAGS)

  string(TOUPPER ${arch} CONFIG_NAME)

  if (${thinlto})
    set(PFP_TEST_USE_THINLTO ${thinlto})
    set(CONFIG_NAME "thinlto-${CONFIG_NAME}")
    list(APPEND PFP_TEST_DEPS LTO)
  endif()
  set(PFP_TEST_USE_THINLTO ${thinlto})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND PFP_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endmacro()

set(PFP_TEST_ARCH ${PFP_SUPPORTED_ARCH})

foreach(arch ${PFP_TEST_ARCH})
  add_pfp_testsuite(${arch} False)
  add_pfp_testsuite(${arch} True)
endforeach()

add_lit_testsuite(check-pfp "Running the PointerFieldProtection tests"
  ${PFP_TESTSUITES}
  DEPENDS ${PFP_TEST_DEPS})

