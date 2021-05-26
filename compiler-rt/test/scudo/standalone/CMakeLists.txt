set(SCUDO_STANDALONE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCUDO_STANDALONE_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SCUDO_STANDALONE_TESTSUITES)
set(SCUDO_STANDALONE_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} scudo_standalone)

foreach(arch ${SCUDO_STANDALONE_SUPPORTED_ARCH})
  set(SCUDO_STANDALONE_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" SCUDO_STANDALONE_TEST_CONFIG_SUFFIX)
  get_target_flags_for_arch(${arch} SCUDO_STANDALONE_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND SCUDO_STANDALONE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

if(COMPILER_RT_INCLUDE_TESTS AND COMPILER_RT_HAS_SCUDO_STANDALONE)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg.py)
  list(APPEND SCUDO_STANDALONE_TEST_DEPS ScudoUnitTests)
  list(APPEND SCUDO_STANDALONE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/unit)
  if (COMPILER_RT_HAS_GWP_ASAN)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/unit/gwp_asan/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/unit/gwp_asan/lit.site.cfg.py)
    list(APPEND SCUDO_STANDALONE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/unit/gwp_asan)
  endif()
endif()

add_lit_testsuite(check-scudo_standalone
  "Running Scudo Standalone tests"
  ${SCUDO_STANDALONE_TESTSUITES}
  DEPENDS ${SCUDO_STANDALONE_TEST_DEPS})

set_target_properties(check-scudo_standalone
  PROPERTIES FOLDER "Compiler-RT Tests")
