set(CSAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(COMPILER_RT_EMULATOR)
  set(CSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS} csan)
  set(CSAN_TEST_TARGET_ARCH ${COMPILER_RT_DEFAULT_TARGET_ARCH})
  set(CSAN_TEST_CONFIG_SUFFIX "-${COMPILER_RT_DEFAULT_TARGET_ARCH}")
  get_test_cc_for_arch(${COMPILER_RT_DEFAULT_TARGET_ARCH}
    CSAN_TEST_TARGET_CC CSAN_TEST_TARGET_CFLAGS)

  if(COMPILER_RT_TARGET_AMDGPU)
    list(APPEND CSAN_TEST_TARGET_CFLAGS
         -Wno-multi-gpu -flto -mcpu=native -nogpulib -stdlib -startfiles)
  else()
    list(APPEND CSAN_TEST_TARGET_CFLAGS
         -Wno-multi-gpu -flto -march=native -nogpulib -stdlib -startfiles)
  endif()
  list(APPEND CSAN_TEST_TARGET_CFLAGS
       --target=${COMPILER_RT_DEFAULT_TARGET_TRIPLE})
  string(REPLACE ";" " " CSAN_TEST_TARGET_CFLAGS "${CSAN_TEST_TARGET_CFLAGS}")

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
  add_lit_testsuite(check-csan "Running GPU ConcurrencySanitizer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CSAN_TEST_DEPS})
  set_target_properties(check-csan PROPERTIES FOLDER "Compiler-RT Tests")
endif()
