set(CTX_PROFILE_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CTX_PROFILE_TESTSUITES)

macro(get_bits_for_arch arch bits)
  if (${arch} MATCHES "x86_64")
    set(${bits} 64)
  else()
    message(FATAL_ERROR "Unexpected target architecture: ${arch}")
  endif()
endmacro()

set(CTX_PROFILE_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} ctx_profile)

foreach(arch ${CTX_PROFILE_SUPPORTED_ARCH})
  set(CTX_PROFILE_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" CTX_PROFILE_TEST_CONFIG_SUFFIX)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py
    )
  list(APPEND CTX_PROFILE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

# Add unit tests.
foreach(arch ${CTX_PROFILE_SUPPORTED_ARCH})
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND CTX_PROFILE_TEST_DEPS CtxProfileUnitTests)
  list(APPEND CTX_PROFILE_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/Unit/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-ctx_profile "Running the Contextual Profiler tests"
  ${CTX_PROFILE_TESTSUITES}
  DEPENDS ${CTX_PROFILE_TEST_DEPS})
set_target_properties(check-ctx_profile PROPERTIES FOLDER "Compiler-RT Misc")
