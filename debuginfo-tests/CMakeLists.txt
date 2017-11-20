# Debug Info tests.  These tests invoke clang to generate programs with
# various types of debug info, and then run those programs under a debugger
# such as GDB or LLDB to verify the results.

if (DEBUGINFO_TESTS_EXCLUDE_FROM_ALL)
  # When run from inside of the clang tree, lit will pick up two different
  # test suites.  It will pick up debuginfo-tests as a standalone test
  # suite since it has a lit.cfg, but it will also run because it will
  # be identified as a subsuite of clang during discovery.  We rely on
  # clang to set this so that this configuration only gets picked up
  # once.
  set(EXCLUDE_FROM_ALL ON)
endif()

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