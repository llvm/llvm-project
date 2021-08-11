if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" LLD_LIBS_DIR "${LLVM_LIBRARY_OUTPUT_INTDIR}")
string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" LLD_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")

llvm_canonicalize_cmake_booleans(
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_LIBXML2
  LLD_DEFAULT_LD_LLD_IS_MINGW
  LLVM_HAVE_LIBXAR
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

set(LLD_TEST_DEPS lld)
if (NOT LLD_BUILT_STANDALONE)
  list(APPEND LLD_TEST_DEPS
    FileCheck count dsymutil llc llvm-ar llvm-as llvm-bcanalyzer llvm-config
    llvm-cvtres llvm-dis llvm-dwarfdump llvm-lib llvm-lipo llvm-mc llvm-nm
    llvm-objcopy llvm-objdump llvm-otool llvm-pdbutil llvm-profdata llvm-readelf
    llvm-readobj llvm-strip llvm-symbolizer not obj2yaml opt split-file yaml2obj
    )
endif()

if (LLVM_INCLUDE_TESTS)
  list(APPEND LLD_TEST_DEPS LLDUnitTests)
endif()

add_lit_testsuite(check-lld "Running lld test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLD_TEST_DEPS}
  )

add_custom_target(lld-test-depends DEPENDS ${LLD_TEST_DEPS})
set_target_properties(lld-test-depends PROPERTIES FOLDER "lld tests")

add_lit_testsuites(LLD ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LLD_TEST_DEPS}
  )

set_target_properties(check-lld PROPERTIES FOLDER "lld tests")

# Add a legacy target spelling: lld-test
add_custom_target(lld-test)
add_dependencies(lld-test check-lld)
set_target_properties(lld-test PROPERTIES FOLDER "lld tests")
