cmake_minimum_required(VERSION 3.20.0)

message("Configuring libsycl tests")

set(LIBSYCL_CXX_COMPILER "${LLVM_BINARY_DIR}/bin/clang++")
set(LIBSYCL_TEST_COMPILER_OPTIONS "" CACHE STRING
    "SYCL compiler options for building libsycl tests")

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

list(APPEND LIBSYCL_TEST_DEPS
    libsycl-toolchain
    FileCheck
    not
)

add_lit_testsuite(check-sycl
  "Running libsycl tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS ${LIBSYCL_TEST_DEPS})
