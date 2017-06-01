set(LLVM_LINK_COMPONENTS
  core
  support
)

if(NOT CLANG_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_clang_tool(clang-import-test
  clang-import-test.cpp
  DEPENDS
  ${tablegen_deps}
  )

set(CLANG_IMPORT_TEST_LIB_DEPS
  clangAST
  clangBasic
  clangCodeGen
  clangFrontend
  clangLex
  clangParse
  )

target_link_libraries(clang-import-test
  ${CLANG_IMPORT_TEST_LIB_DEPS}
  )
