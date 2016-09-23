set(LLVM_SOURCE_DIR "${LLVM_MAIN_SRC_DIR}")
set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_BUILD_MODE "%(build_mode)s")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}/%(build_config)s")
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/%(build_config)s")
set(CLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

set(LLD_TEST_DEPS
  FileCheck not llvm-ar llvm-as llvm-dis llvm-nm llc
  lld llvm-config llvm-objdump llvm-readobj yaml2obj obj2yaml
  llvm-mc llvm-nm llvm-lib llvm-pdbdump
  )
if (LLVM_INCLUDE_TESTS)
  set(LLD_TEST_DEPS ${LLD_TEST_DEPS} LLDUnitTests)
endif()

set(LLD_TEST_PARAMS
  lld_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-lld "Running lld test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS lld_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
       lld_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${LLD_TEST_DEPS}
  )

set_target_properties(check-lld PROPERTIES FOLDER "lld tests")

# Add a legacy target spelling: lld-test
add_custom_target(lld-test)
add_dependencies(lld-test check-lld)
set_target_properties(lld-test PROPERTIES FOLDER "lld tests")
