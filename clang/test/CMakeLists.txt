# Test runner infrastructure for Clang. This configures the Clang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
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
  )
  
if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND CLANG_TEST_DEPS
    clang-check
    )
endif()

if (CLANG_ENABLE_ARCMT)
  list(APPEND CLANG_TEST_DEPS
    arcmt-test
    c-arcmt-test
  )
endif ()

if (ENABLE_CLANG_EXAMPLES)
  list(APPEND CLANG_TEST_DEPS
    AnnotateFunctions
    clang-interpreter
    PrintFunctionNames
    )
endif ()

if (ENABLE_CLANG_STATIC_ANALYZER AND ENABLE_CLANG_EXAMPLES)
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
