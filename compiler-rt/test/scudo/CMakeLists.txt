set(SCUDO_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SCUDO_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(SCUDO_TESTSUITES)

set(SCUDO_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND SCUDO_TEST_DEPS scudo)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

set(SCUDO_TEST_ARCH ${SCUDO_SUPPORTED_ARCH})
foreach(arch ${SCUDO_TEST_ARCH})
  set(SCUDO_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" SCUDO_TEST_CONFIG_SUFFIX)
  
  if(ANDROID OR ${arch} MATCHES "arm|aarch64")
    # This is only true if we are cross-compiling.
    # Build all tests with host compiler and use host tools.
    set(SCUDO_TEST_TARGET_CFLAGS ${COMPILER_RT_TEST_COMPILER_CFLAGS})
  else()
    get_target_flags_for_arch(${arch} SCUDO_TEST_TARGET_CFLAGS)
    string(REPLACE ";" " " SCUDO_TEST_TARGET_CFLAGS "${SCUDO_TEST_TARGET_CFLAGS}")
  endif()

  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND SCUDO_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-scudo "Running the Scudo Hardened Allocator tests"
  ${SCUDO_TESTSUITES}
  DEPENDS ${SCUDO_TEST_DEPS})
set_target_properties(check-scudo PROPERTIES FOLDER "Compiler-RT Misc")
