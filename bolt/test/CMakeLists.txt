llvm_canonicalize_cmake_booleans(
  BOLT_ENABLE_RUNTIME
  )

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

set(BOLT_TEST_PARAMS
  bolt_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

if (BOLT_CLANG_EXE AND NOT "clang" IN_LIST LLVM_ENABLE_PROJECTS)
  add_executable(clang IMPORTED GLOBAL)
  set_property(TARGET clang PROPERTY IMPORTED_LOCATION "${BOLT_CLANG_EXE}")
endif()

if (BOLT_LLD_EXE AND NOT "lld" IN_LIST LLVM_ENABLE_PROJECTS)
  add_executable(lld IMPORTED GLOBAL)
  set_property(TARGET lld PROPERTY IMPORTED_LOCATION "${BOLT_LLD_EXE}")
endif()

list(APPEND BOLT_TEST_DEPS
  clang
  count
  FileCheck
  llc
  lld
  llvm-config
  llvm-bolt
  llvm-boltdiff
  llvm-bolt-heatmap
  llvm-bat-dump
  llvm-dwarfdump
  llvm-dwp
  llvm-mc
  llvm-nm
  llvm-objdump
  llvm-readelf
  llvm-readobj
  llvm-strings
  llvm-strip
  llvm-objcopy
  merge-fdata
  not
  perf2bolt
  split-file
  yaml2obj
  )

add_custom_target(bolt-test-depends DEPENDS ${BOLT_TEST_DEPS})
set_target_properties(bolt-test-depends PROPERTIES FOLDER "BOLT")

add_lit_testsuite(check-bolt "Running the BOLT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${BOLT_TEST_PARAMS}
  DEPENDS ${BOLT_TEST_DEPS}
  ARGS ${BOLT_TEST_EXTRA_ARGS}
  )
set_target_properties(check-bolt PROPERTIES FOLDER "BOLT")

add_lit_testsuites(BOLT ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${BOLT_TEST_PARAMS}
  DEPENDS ${BOLT_TEST_DEPS}
)
