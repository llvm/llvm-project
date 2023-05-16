set(CLANGD_TEST_DEPS
  clangd
  ClangdTests
  clangd-indexer
  # No tests for it, but we should still make sure they build.
  dexp
  )

if(CLANGD_BUILD_XPC)
  list(APPEND CLANGD_TEST_DEPS clangd-xpc-test-client)
  list(APPEND CLANGD_TEST_DEPS ClangdXpcUnitTests)
endif()

if(CLANGD_ENABLE_REMOTE)
  list(APPEND CLANGD_TEST_DEPS clangd-index-server clangd-index-server-monitor)
endif()

foreach(dep FileCheck count not llvm-config)
  if(TARGET ${dep})
    list(APPEND CLANGD_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

# Copy an empty compile_commands.json to override the compile_commands.json
# in the top level build directory.  Or if a clangd test involves creating a
# temporary source file in the build directory and run clangd to check it,
# it can pick up unrecognizable command options when LLVM is built with
# another compiler or a different version of Clang.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
)

add_lit_testsuite(check-clangd "Running the Clangd regression tests"
  # clangd doesn't put unittest configs in test/unit like every other project.
  # Because of that, this needs to pass two folders here, while every other
  # project only needs to pass CMAKE_CURRENT_BINARY_DIR.
  ${CMAKE_CURRENT_BINARY_DIR}/../unittests;${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANGD_TEST_DEPS})
