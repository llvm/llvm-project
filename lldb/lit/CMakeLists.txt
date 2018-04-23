# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

if (NOT LLDB_TEST_USE_CUSTOM_C_COMPILER)
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TEST_C_COMPILER ${LLDB_TEST_C_COMPILER})
endif ()

if (NOT LLDB_TEST_USE_CUSTOM_CXX_COMPILER)
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TEST_CXX_COMPILER ${LLDB_TEST_CXX_COMPILER})
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_LIBS_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

if(BUILD_SHARED_LIBS)
  set(ENABLE_SHARED 1)
else()
  set(ENABLE_SHARED 0)
endif(BUILD_SHARED_LIBS)

# the value is not canonicalized within LLVM
llvm_canonicalize_cmake_booleans(
  LLVM_ENABLE_ZLIB)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

set(LLDB_TEST_DEPS
  LLDBUnitTests
  dsymutil
  lldb
  lldb-test
  llvm-config
  llvm-mc
  llvm-objcopy
  )

if(NOT LLDB_BUILT_STANDALONE)
  list(APPEND LLDB_TEST_DEPS FileCheck not yaml2obj)
endif()
  
# lldb-server is not built on every platform.
if (TARGET lldb-server)
  list(APPEND LLDB_TEST_DEPS lldb-server)
endif()
  
if(APPLE)
  list(APPEND LLDB_TEST_DEPS debugserver)
endif()

if(TARGET clang)
  list(APPEND LLDB_TEST_DEPS clang)
endif()

set(LLDB_TEST_PARAMS
  lldb_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-lldb-lit "Running lldb lit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS lldb_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
       lldb_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${LLDB_TEST_DEPS}
  )

set_target_properties(check-lldb-lit PROPERTIES FOLDER "LLDB tests")

# If we're building with an in-tree clang, then list clang as a dependency
# to run tests.
if (TARGET clang)
  add_dependencies(check-lldb-lit clang)
endif()

add_lit_testsuites(LLDB ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS lldb_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
         lldb_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${LLDB_TEST_DEPS}
  )
