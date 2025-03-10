# Test runner infrastructure for Clang. This configures the Clang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  CLANG_BUILD_EXAMPLES
  CLANG_BUILT_STANDALONE
  CLANG_DEFAULT_PIE_ON_LINUX
  CLANG_ENABLE_STATIC_ANALYZER
  CLANG_PLUGIN_SUPPORT
  CLANG_SPAWN_CC1
  CLANG_ENABLE_CIR
  CLANG_ENABLE_OBJC_REWRITER
  ENABLE_BACKTRACES
  LLVM_BUILD_EXAMPLES
  LLVM_BYE_LINK_INTO_TOOLS
  LLVM_ENABLE_PLUGINS
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_ZSTD
  LLVM_ENABLE_PER_TARGET_RUNTIME_DIR
  LLVM_ENABLE_THREADS
  LLVM_ENABLE_REVERSE_ITERATION
  LLVM_WITH_Z3
  PPC_LINUX_DEFAULT_IEEELONGDOUBLE
  LLVM_TOOL_LLVM_DRIVER_BUILD
  LLVM_INCLUDE_SPIRV_TOOLS_TESTS
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
  "CURRENT_TOOLS_DIR"
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
  clang-fuzzer-dictionary
  clang-resource-headers
  clang-format
  clang-tblgen
  clang-offload-bundler
  clang-import-test
  clang-refactor
  clang-diff
  clang-installapi
  clang-scan-deps
  clang-linker-wrapper
  clang-nvlink-wrapper
  clang-offload-bundler
  clang-offload-packager
  clang-sycl-linker
  diagtool
  hmaptool
  )

if(CLANG_ENABLE_CIR)
  list(APPEND CLANG_TEST_DEPS
    cir-opt
    )
endif()

if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND CLANG_TEST_DEPS
    clang-check
    clang-extdef-mapping
    )
endif()

if(CLANG_BUILD_EXAMPLES AND CLANG_PLUGIN_SUPPORT)
  list(APPEND CLANG_TEST_DEPS
    Attribute
    AnnotateFunctions
    CallSuperAttr
    PluginsOrder
    PrintFunctionNames
    LLVMPrintFunctionNames
    )
endif ()

if(LLVM_INCLUDE_SPIRV_TOOLS_TESTS)
  list(APPEND CLANG_TEST_DEPS
    spirv-dis
    spirv-val
    spirv-as
    spirv-link
    )
endif()    

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
    llvm-link
    llvm-lto2
    llvm-mc
    llvm-modextract
    llvm-nm
    llvm-objcopy
    llvm-objdump
    llvm-pdbutil
    llvm-profdata
    llvm-rc
    llvm-readelf
    llvm-readobj
    llvm-readtapi
    llvm-strip
    llvm-symbolizer
    llvm-windres
    obj2yaml
    opt
    split-file
    yaml2obj
    )

  if(TARGET llvm-lto)
    list(APPEND CLANG_TEST_DEPS llvm-lto)
  endif()
endif()

if(CLANG_ENABLE_STATIC_ANALYZER)
  if(CLANG_PLUGIN_SUPPORT AND LLVM_ENABLE_PLUGINS) # Determine if we built them
    list(APPEND CLANG_TEST_DEPS
      SampleAnalyzerPlugin
      CheckerDependencyHandlingAnalyzerPlugin
      CheckerOptionHandlingAnalyzerPlugin
      )
  endif()
endif()

if (HAVE_CLANG_REPL_SUPPORT)
  list(APPEND CLANG_TEST_DEPS
    clang-repl
    )
endif()

# Copy gen_ast_dump_json_test.py to the clang build dir. This allows invoking
# it without having to pass the --clang= argument
configure_file(AST/gen_ast_dump_json_test.py
  ${CLANG_BINARY_DIR}/bin/gen_ast_dump_json_test.py COPYONLY)

add_custom_target(clang-test-depends DEPENDS ${CLANG_TEST_DEPS})
set_target_properties(clang-test-depends PROPERTIES FOLDER "Clang/Tests")

add_lit_testsuite(check-clang "Running the Clang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  PARAMS ${CLANG_TEST_PARAMS}
  DEPENDS ${CLANG_TEST_DEPS}
  ARGS ${CLANG_TEST_EXTRA_ARGS}
  )

add_lit_testsuites(CLANG ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${CLANG_TEST_PARAMS}
  DEPENDS ${CLANG_TEST_DEPS}
  FOLDER "Clang tests/Suites"
)

# Add a legacy target spelling: clang-test
add_custom_target(clang-test)
add_dependencies(clang-test check-clang)
set_target_properties(clang-test PROPERTIES FOLDER "Clang/Tests")

# FIXME: This logic can be removed once all buildbots have moved
# debuginfo-test from clang/test to llvm/projects or monorepo.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/debuginfo-tests)
  message(WARNING "Including debuginfo-tests in clang/test is deprecated.  Move to llvm/projects or use monorepo.")
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/debuginfo-tests/CMakeLists.txt)
    add_subdirectory(debuginfo-tests)
  endif()
endif()
