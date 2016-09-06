set(XRAY_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(XRAY_TESTSUITES)

set(XRAY_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})

if(NOT COMPILER_RT_STANDALONE_BUILD AND COMPILER_RT_BUILD_XRAY AND
   COMPILER_RT_HAS_XRAY)
  list(APPEND XRAY_TEST_DEPS xray)
endif()

set(XRAY_TEST_ARCH ${XRAY_SUPPORTED_ARCH})
foreach(arch ${XRAY_TEST_ARCH})
  set(XRAY_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" XRAY_TEST_CONFIG_SUFFIX)

  if(ANDROID OR ${arch} MATCHES "arm|aarch64")
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(XRAY_TEST_TARGET_CC ${COMPILER_RT_TEST_COMPILER})
    set(XRAY_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
  else()
    get_target_flags_for_arch(${arch} XRAY_TEST_TARGET_CFLAGS)
    string(REPLACE ";" " " XRAY_TEST_TARGET_CFLAGS "${XRAY_TEST_TARGET_CFLAGS}")
  endif()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND XRAY_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-xray "Running the XRay tests"
  ${XRAY_TESTSUITES}
	DEPENDS ${XRAY_TEST_DEPS})
set_target_properties(check-xray PROPERTIES FOLDER "Compiler-RT Misc")
