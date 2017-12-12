# Test runner infrastructure for Clang. This configures the Clang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

if(CLANG_BUILT_STANDALONE)
  # Set HAVE_LIBZ according to recorded LLVM_ENABLE_ZLIB value. This
  # value is forced to 0 if zlib was not found, so it is fine to use it
  # instead of HAVE_LIBZ (not recorded).
  if(LLVM_ENABLE_ZLIB)
    set(HAVE_LIBZ 1)
  endif()
endif()

llvm_canonicalize_cmake_booleans(
  CLANG_BUILD_EXAMPLES
  CLANG_ENABLE_ARCMT
  CLANG_ENABLE_STATIC_ANALYZER
  ENABLE_BACKTRACES
  HAVE_LIBZ)

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

option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
if(CLANG_TEST_USE_VG)
  set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND CLANG_TEST_DEPS
  clang clang-headers
  clang-format
  c-index-test diagtool
  clang-tblgen
  clang-offload-bundler
  clang-import-test
  clang-rename
  clang-refactor
  clang-diff
  )
  
if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND CLANG_TEST_DEPS
    clang-check
    clang-func-mapping
    )
endif()

if (CLANG_ENABLE_ARCMT)
  list(APPEND CLANG_TEST_DEPS
    arcmt-test
    c-arcmt-test
  )
endif ()

if (CLANG_BUILD_EXAMPLES)
  list(APPEND CLANG_TEST_DEPS
    AnnotateFunctions
    clang-interpreter
    PrintFunctionNames
    )
endif ()

if (CLANG_ENABLE_STATIC_ANALYZER AND CLANG_BUILD_EXAMPLES)
  list(APPEND CLANG_TEST_DEPS
    SampleAnalyzerPlugin
    )
endif ()

set(CLANG_TEST_PARAMS
  clang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

if( NOT CLANG_BUILT_STANDALONE )
  list(APPEND CLANG_TEST_DEPS
    llvm-config
    FileCheck count not
    llc
    llvm-bcanalyzer
    llvm-cat
    llvm-dis
    llvm-modextract
    llvm-nm
    llvm-objdump
    llvm-profdata
    llvm-readobj
    llvm-symbolizer
    opt
    )

  if(TARGET llvm-lto)
    list(APPEND CLANG_TEST_DEPS llvm-lto)
  endif()
endif()

add_custom_target(clang-test-depends DEPENDS ${CLANG_TEST_DEPS})
set_target_properties(clang-test-depends PROPERTIES FOLDER "Clang tests")

add_lit_testsuite(check-clang "Running the Clang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${CLANG_TEST_PARAMS}
  DEPENDS ${CLANG_TEST_DEPS}
  ARGS ${CLANG_TEST_EXTRA_ARGS}
  )
set_target_properties(check-clang PROPERTIES FOLDER "Clang tests")

add_lit_testsuites(CLANG ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${CLANG_TEST_PARAMS}
  DEPENDS ${CLANG_TEST_DEPS}
)

# Add a legacy target spelling: clang-test
add_custom_target(clang-test)
add_dependencies(clang-test check-clang)
set_target_properties(clang-test PROPERTIES FOLDER "Clang tests")

# FIXME: This logic can be removed once all buildbots have moved
# debuginfo-test from clang/test to llvm/projects or monorepo.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/debuginfo-tests)
  message(WARNING "Including debuginfo-tests in clang/test is deprecated.  Move to llvm/projects or use monorepo.")
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/debuginfo-tests/CMakeLists.txt)
    add_subdirectory(debuginfo-tests)
  endif()
endif()
