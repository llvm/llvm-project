# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

# LLVM_BUILD_MODE is used in lit.site.cfg
if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LLDB_IS_64_BITS 1)
endif()

get_property(LLDB_DOTEST_ARGS GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY)
set(dotest_args_replacement ${LLVM_BUILD_MODE})

set(LLDB_TEST_MODULE_CACHE_LLDB "${LLDB_TEST_BUILD_DIRECTORY}/module-cache-lldb" CACHE PATH "The Clang module cache used by the Clang embedded in LLDB while running tests.")
set(LLDB_TEST_MODULE_CACHE_CLANG "${LLDB_TEST_BUILD_DIRECTORY}/module-cache-clang" CACHE PATH "The Clang module cache used by the Clang while building tests.")

if(LLDB_BUILT_STANDALONE)
  # In paths to our build-tree, replace CMAKE_CFG_INTDIR with our configuration name placeholder.
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} config_runtime_output_dir ${LLVM_RUNTIME_OUTPUT_INTDIR})
  string(REPLACE ${LLVM_RUNTIME_OUTPUT_INTDIR} ${config_runtime_output_dir} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")

  # Remaining ones must be paths to the provided LLVM build-tree.
  if(LLVM_CONFIGURATION_TYPES)
    # LLDB uses single-config; LLVM multi-config; pick one and prefer Release types.
    # Otherwise, if both use multi-config the default is fine.
    if(NOT CMAKE_CONFIGURATION_TYPES)
      if(RelWithDebInfo IN_LIST LLVM_CONFIGURATION_TYPES)
        set(dotest_args_replacement RelWithDebInfo)
      elseif(Release IN_LIST LLVM_CONFIGURATION_TYPES)
        set(dotest_args_replacement Release)
      else()
        list(GET LLVM_CONFIGURATION_TYPES 0 dotest_args_replacement)
      endif()
    endif()
  else()
    # Common case: LLVM used a single-configuration generator like Ninja.
    set(dotest_args_replacement ".")
  endif()
endif()

string(REPLACE ${CMAKE_CFG_INTDIR} ${dotest_args_replacement} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_LIBS_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

add_lldb_test_dependency(
  lit-cpuid
  llc
  lli
  llvm-config
  llvm-dwarfdump
  llvm-nm
  llvm-mc
  llvm-objcopy
  llvm-readobj
  llvm-strip
  )

if(TARGET lld)
  add_lldb_test_dependency(lld)
endif()

# the value is not canonicalized within LLVM
llvm_canonicalize_cmake_booleans(
  LLDB_DISABLE_PYTHON
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_SHARED_LIBS
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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit-lldb-init.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init)

if(NOT LLDB_BUILT_STANDALONE)
  add_lldb_test_dependency(
    FileCheck
    count
    not
  )
endif()

add_lit_testsuite(check-lldb-lit "Running lldb lit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS lldb-test-deps
  )

set_target_properties(check-lldb-lit PROPERTIES FOLDER "lldb tests")

# If we're building with an in-tree clang, then list clang as a dependency
# to run tests.
if (TARGET clang)
  add_dependencies(check-lldb-lit clang)
endif()

add_lit_testsuites(LLDB
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS lldb-test-deps
  )
