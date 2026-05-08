set(UBSAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(UBSAN_TEST_ARCH ${UBSAN_SUPPORTED_ARCH})
if(APPLE)
  darwin_filter_host_archs(UBSAN_SUPPORTED_ARCH UBSAN_TEST_ARCH)
endif()

set(UBSAN_TESTSUITES)
set(UBSAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND UBSAN_TEST_DEPS ubsan-minimal)

foreach(arch ${UBSAN_TEST_ARCH})
  set(UBSAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" UBSAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} UBSAN_TEST_TARGET_CC UBSAN_TEST_TARGET_CFLAGS)

  if(COMPILER_RT_GPU_BUILD)
    if("${arch}" MATCHES "amdgcn")
      list(APPEND UBSAN_TEST_TARGET_CFLAGS
           -Wno-multi-gpu -flto -mcpu=native -nogpulib -startfiles -stdlib)
    elseif("${arch}" MATCHES "nvptx")
      list(APPEND UBSAN_TEST_TARGET_CFLAGS
           -Wno-multi-gpu -flto -march=native -nogpulib -startfiles -stdlib)
    endif()
    list(APPEND UBSAN_TEST_TARGET_CFLAGS --target=${COMPILER_RT_DEFAULT_TARGET_TRIPLE})
    string(REPLACE ";" " " UBSAN_TEST_TARGET_CFLAGS "${UBSAN_TEST_TARGET_CFLAGS}")
  endif()

  set(CONFIG_NAME ${arch})
  if(${arch} IN_LIST CFI_SUPPORTED_ARCH)
    set(UBSAN_TEST_HAS_CFI TRUE)
  else()
    set(UBSAN_TEST_HAS_CFI FALSE)
  endif()
  pythonize_bool(UBSAN_TEST_HAS_CFI)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND UBSAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-ubsan-minimal "Running UndefinedBehaviorSanitizerMinimal tests"
  ${UBSAN_TESTSUITES}
  DEPENDS ${UBSAN_TEST_DEPS})
