set(GWP_ASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GWP_ASAN_LIT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(GWP_ASAN_TESTSUITES)

set(GWP_ASAN_UNITTEST_DEPS)
set(GWP_ASAN_TEST_DEPS
  ${SANITIZER_COMMON_LIT_TEST_DEPS}
  gwp_asan
  scudo)

# Longstanding issues in the Android test runner means that compiler-rt unit
# tests don't work on Android due to libc++ link-time issues. Looks like the
# exported libc++ from the Android NDK is x86-64, even though it's part of the
# ARM[64] toolchain... See similar measures for ASan and sanitizer-common that
# disable unit tests for Android.
if (COMPILER_RT_INCLUDE_TESTS AND COMPILER_RT_HAS_GWP_ASAN AND NOT ANDROID)
  list(APPEND GWP_ASAN_TEST_DEPS GwpAsanUnitTests)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg)
  add_lit_testsuite(check-gwp_asan-unit "Running GWP-ASan unit tests"
    ${CMAKE_CURRENT_BINARY_DIR}/unit
    DEPENDS ${GWP_ASAN_TEST_DEPS})
  set_target_properties(check-gwp_asan-unit PROPERTIES FOLDER
    "Compiler-RT Tests")
    list(APPEND GWP_ASAN_TEST_DEPS check-gwp_asan-unit)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

foreach(arch ${GWP_ASAN_SUPPORTED_ARCH})
  set(GWP_ASAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}" GWP_ASAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} GWP_ASAN_TEST_TARGET_CC GWP_ASAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg)
  list(APPEND GWP_ASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-gwp_asan "Running the GWP-ASan tests"
  ${GWP_ASAN_TESTSUITES}
  DEPENDS ${GWP_ASAN_TEST_DEPS})
set_target_properties(check-gwp_asan PROPERTIES FOLDER "Compiler-RT Misc")
