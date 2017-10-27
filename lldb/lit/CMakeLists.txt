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

set(LLDB_TEST_DEPS
  LLDBUnitTests
  lldb
  )

if(NOT LLDB_BUILT_STANDALONE)
  list(APPEND LLDB_TEST_DEPS FileCheck not)
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

add_lit_testsuites(LLDB ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS lldb_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
         lldb_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${LLDB_TEST_DEPS}
  )
