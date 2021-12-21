# Test runner infrastructure for Clang. This configures the Clang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

llvm_canonicalize_cmake_booleans(
  CLANG_BUILD_EXAMPLES
  CLANG_DEFAULT_PIE_ON_LINUX
  CLANG_ENABLE_ARCMT
  CLANG_ENABLE_STATIC_ANALYZER
  CLANG_SPAWN_CC1
  ENABLE_BACKTRACES
  LLVM_ENABLE_NEW_PASS_MANAGER
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_PER_TARGET_RUNTIME_DIR
  LLVM_ENABLE_PLUGINS
  LLVM_ENABLE_THREADS
  LLVM_WITH_Z3
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "SHLIBDIR"
  "LLVM_LIT_TOOLS_DIR"
  "LLVM_EXTERNAL_LIT"
  "CLANG_BINARY_DIR"
  "CLANG_SOURCE_DIR"
  "CLANG_TOOLS_DIR"
  "CMAKE_LIBRARY_OUTPUT_DIRECTORY"
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "CLANG_BINARY_DIR"
  "SHLIBDIR"
  )

option(CLANG_TEST_USE_VG "Run Clang tests under Valgrind" OFF)
if(CLANG_TEST_USE_VG)
  set(CLANG_TEST_EXTRA_ARGS ${CLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND CLANG_TEST_DEPS
  apinotes-test
  c-index-test
  clang
  clang-resource-headers
  clang-format
  clang-tblgen
  clang-offload-bundler
  clang-import-test
  clang-rename
  clang-refactor
  clang-repl
  clang-diff
  clang-scan-deps
  diagtool
  hmaptool
  )
  
if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND CLANG_TEST_DEPS
    clang-check
    clang-extdef-mapping
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
    Attribute
    AnnotateFunctions
    CallSuperAttr
    PluginsOrder
    PrintFunctionNames
    )
endif ()

set(CLANG_TEST_PARAMS
  USE_Z3_SOLVER=0
  )

if( NOT CLANG_BUILT_STANDALONE )
  list(APPEND CLANG_TEST_DEPS
    llvm-config
    FileCheck count not
    llc
    llvm-ar
    llvm-as
    llvm-bcanalyzer
    llvm-cat
    llvm-cxxfilt
    llvm-dis
    llvm-dwarfdump
    llvm-ifs
    llvm-lto2
    llvm-modextract
    llvm-nm
    llvm-objcopy
    llvm-objdump
    llvm-pdbutil
    llvm-profdata
    llvm-rc
    llvm-readelf
    llvm-readobj
    llvm-strip
    llvm-symbolizer
    llvm-windres
    opt
    split-file
    yaml2obj
    )

  if(TARGET llvm-lto)
    list(APPEND CLANG_TEST_DEPS llvm-lto)
  endif()
endif()

if (CLANG_ENABLE_STATIC_ANALYZER)
  if (LLVM_ENABLE_PLUGINS)
    list(APPEND CLANG_TEST_DEPS
      SampleAnalyzerPlugin
      CheckerDependencyHandlingAnalyzerPlugin
      CheckerOptionHandlingAnalyzerPlugin
      )
  endif()
endif()

# Copy gen_ast_dump_json_test.py to the clang build dir. This allows invoking
# it without having to pass the --clang= argument
configure_file(AST/gen_ast_dump_json_test.py
  ${CLANG_BINARY_DIR}/bin/gen_ast_dump_json_test.py COPYONLY)

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
