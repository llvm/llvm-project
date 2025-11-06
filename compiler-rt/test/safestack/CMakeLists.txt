set(SAFESTACK_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAFESTACK_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SAFESTACK_TESTSUITES)
set(SAFESTACK_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND SAFESTACK_TEST_DEPS safestack)

macro(add_safestack_testsuite test_mode sanitizer arch)
  set(SAFESTACK_LIT_TEST_MODE "${test_mode}")
  set(CONFIG_NAME ${SAFESTACK_LIT_TEST_MODE})

  if(NOT COMPILER_RT_STANDALONE_BUILD)
    # Some tests require LTO, so add a dependency on the relevant LTO plugin.
    if(LLVM_ENABLE_PIC)
      if(LLVM_BINUTILS_INCDIR)
        list(APPEND SAFESTACK_TEST_DEPS LLVMgold)
      endif()
      if(APPLE)
        list(APPEND SAFESTACK_TEST_DEPS LTO)
      endif()
    endif()
  endif()
  set(CONFIG_NAME ${CONFIG_NAME}-${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND SAFESTACK_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endmacro()

set(SAFESTACK_TEST_ARCH ${SAFESTACK_SUPPORTED_ARCH})

foreach(arch ${SAFESTACK_TEST_ARCH})
  set(SAFESTACK_TEST_TARGET_ARCH ${arch})
  get_test_cc_for_arch(${arch} SAFESTACK_TEST_TARGET_CC SAFESTACK_TEST_TARGET_CFLAGS)
  add_safestack_testsuite("Standalone" safestack ${arch})
endforeach()

add_lit_testsuite(check-safestack "Running the SafeStack tests"
  ${SAFESTACK_TESTSUITES}
  DEPENDS ${SAFESTACK_TEST_DEPS})
