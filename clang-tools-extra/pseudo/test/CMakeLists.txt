# Set CLANG_TOOLS_DIR to buildtree/bin, or buildtree/%(build_mode)s/bin if the
# location is dynamic. The latter must be interpolated by lit configs.
# FIXME: this is duplicated in many places.
if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

set(CLANG_PSEUDO_TEST_DEPS
  clang-pseudo
  ClangPseudoTests
  )

foreach(dep FileCheck not)
  if(TARGET ${dep})
    list(APPEND CLANG_PSEUDO_TEST_DEPS ${dep})
  endif()
endforeach()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

add_lit_testsuite(check-clang-pseudo "Running the clang-pseudo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANG_PSEUDO_TEST_DEPS})
