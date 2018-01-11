set(CSI_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND CSI_TEST_DEPS csi)
endif()

# Add a dependency on the LTO plugin.
if(LLVM_BINUTILS_INCDIR)
  list(APPEND CSI_TEST_DEPS LLVMgold)
endif()

set(CSI_TESTSUITES)

set(CSI_TEST_ARCH ${CSI_SUPPORTED_ARCH})

set(CSI_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach(arch ${CSI_TEST_ARCH})
  set(CSI_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" CSI_TEST_CONFIG_SUFFIX)
  get_target_flags_for_arch(${arch} CSI_TEST_TARGET_CFLAGS)
  string(REPLACE ";" " " CSI_TEST_TARGET_CFLAGS "${CSI_TEST_TARGET_CFLAGS}")

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND CSI_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-csi "Running ComprehensiveStaticInstrumentation tests"
  ${CSI_TESTSUITES}
  DEPENDS ${CSI_TEST_DEPS})
set_target_properties(check-csi PROPERTIES FOLDER "Csi tests")
