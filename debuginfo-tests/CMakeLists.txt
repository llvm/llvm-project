# Debug Info tests.  These tests invoke clang to generate programs with
# various types of debug info, and then run those programs under a debugger
# such as GDB or LLDB to verify the results.

set(DEBUGINFO_TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEBUGINFO_TESTS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(DEBUGINFO_TEST_DEPS
  clang
  llvm-config
  FileCheck
  count
  not
  )

get_target_property(CLANG_SOURCE_DIR clang SOURCE_DIR)

if (TARGET lld)
  set(DEBUGINFO_TESTS_HAS_LLD 1)
  list(APPEND DEBUGINFO_TEST_DEPS lld)
  get_target_property(LLD_SOURCE_DIR lld SOURCE_DIR)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-debuginfo "Running debug info integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DEBUGINFO_TEST_DEPS}
  )

set_target_properties(check-debuginfo PROPERTIES FOLDER "Debug info tests")