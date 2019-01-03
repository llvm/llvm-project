# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LLDB_IS_64_BITS 1)
endif()

if (NOT LLDB_TEST_USE_CUSTOM_C_COMPILER)
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TEST_C_COMPILER ${LLDB_TEST_C_COMPILER})
endif ()

if (NOT LLDB_TEST_USE_CUSTOM_CXX_COMPILER)
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TEST_CXX_COMPILER ${LLDB_TEST_CXX_COMPILER})
endif ()

get_property(LLDB_DOTEST_ARGS GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY)

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_LIBS_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")

list(APPEND LLDB_TEST_DEPS
  count
  LLDBUnitTests
  dsymutil
  llc
  lldb
  lldb-test
  llvm-config
  llvm-mc
  llvm-objcopy
  )

if(TARGET lld)
  list(APPEND LLDB_TEST_DEPS lld)
  set(LLDB_HAVE_LLD 1)
else()
  set(LLDB_HAVE_LLD 0)
endif()

if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

# the value is not canonicalized within LLVM
llvm_canonicalize_cmake_booleans(
  LLDB_DISABLE_PYTHON
  LLVM_ENABLE_ZLIB
  LLDB_IS_64_BITS)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Suite/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Suite/lit.site.cfg)

if(NOT LLDB_BUILT_STANDALONE)
  list(APPEND LLDB_TEST_DEPS
    FileCheck
    not
    )
endif()

add_lit_testsuite(check-lldb-lit "Running lldb lit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLDB_TEST_DEPS}
  )

set_target_properties(check-lldb-lit PROPERTIES FOLDER "LLDB tests")

# If we're building with an in-tree clang, then list clang as a dependency
# to run tests.
if (TARGET clang)
  add_dependencies(check-lldb-lit clang)
endif()

add_lit_testsuites(LLDB
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${LLDB_TEST_DEPS}
  )
