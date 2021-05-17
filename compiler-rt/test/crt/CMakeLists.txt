set(CRT_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CRT_TESTSUITES)

set(CRT_TEST_DEPS)
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND CRT_TEST_DEPS crt)
endif()
if(NOT COMPILER_RT_STANDALONE_BUILD AND NOT LLVM_RUNTIMES_BUILD)
  # Use LLVM utils and Clang from the same build tree.
  list(APPEND CRT_TEST_DEPS
    clang clang-resource-headers FileCheck not llvm-config)
endif()

set(CRT_TEST_ARCH ${CRT_SUPPORTED_ARCH})
if (COMPILER_RT_BUILD_CRT AND COMPILER_RT_HAS_CRT)
  foreach(arch ${CRT_TEST_ARCH})
    set(CRT_TEST_TARGET_ARCH ${arch})
    string(TOLOWER "-${arch}-${OS_NAME}" CRT_TEST_CONFIG_SUFFIX)
    get_test_cc_for_arch(${arch} CRT_TEST_TARGET_CC CRT_TEST_TARGET_CFLAGS)
    string(TOUPPER ${arch} ARCH_UPPER_CASE)
    set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
    list(APPEND CRT_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
  endforeach()
endif()

add_lit_testsuite(check-crt "Running the CRT tests"
  ${CRT_TESTSUITES}
  DEPENDS ${CRT_TEST_DEPS})
set_target_properties(check-crt PROPERTIES FOLDER "Compiler-RT Misc")
