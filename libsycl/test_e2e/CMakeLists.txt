cmake_minimum_required(VERSION 3.20.0)

message("Configuring libsycl End-to-End Tests")

set(LIBSYCL_CXX_COMPILER "${LLVM_BINARY_DIR}/bin/clang++")
set(LIBSYCL_E2E_CXX_FLAGS "" CACHE STRING
    "Flags passed to clang++ when building SYCL end-to-end tests")

if(NOT LIBSYCL_TEST_E2E_TARGETS)
  set(LIBSYCL_TEST_E2E_TARGETS "all")
endif()

if(NOT DEFINED LEVEL_ZERO_LIBS_DIR AND NOT DEFINED LEVEL_ZERO_INCLUDE_DIR)
    find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h)
    if(LEVEL_ZERO_INCLUDE_DIR)
      find_library(LEVEL_ZERO_LIBS_DIR NAMES ze_loader)
    endif()
endif()

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

list(APPEND LIBSYCL_E2E_TEST_DEPS
    libsycl-toolchain
    FileCheck
    not
)

add_lit_testsuite(check-sycl-e2e
  "Running libsycl End-to-End tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS ${LIBSYCL_E2E_TEST_DEPS})
