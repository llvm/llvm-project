# Set CLANG_TOOLS_DIR to buildtree/bin, or buildtree/%(build_mode)s/bin if the
# location is dynamic. The latter must be interpolated by lit configs.
# FIXME: this is duplicated in many places.
if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

set(CLANGD_TEST_DEPS
  clangd
  ClangdTests
  # No tests for these, but we should still make sure they build.
  clangd-indexer
  dexp
  )

if(CLANGD_BUILD_XPC)
  list(APPEND CLANGD_TEST_DEPS clangd-xpc-test-client)
  list(APPEND CLANGD_TEST_DEPS ClangdXpcUnitTests)
endif()

foreach(dep FileCheck count not)
  if(TARGET ${dep})
    list(APPEND CLANGD_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

add_lit_testsuite(check-clangd "Running the Clangd regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}/../unittests;${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANGD_TEST_DEPS})
