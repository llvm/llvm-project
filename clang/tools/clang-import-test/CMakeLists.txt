set(LLVM_LINK_COMPONENTS
  Core
  Support
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
  clangDriver
  clangFrontend
  clangLex
  clangParse
  )

target_link_libraries(clang-import-test
  PRIVATE
  ${CLANG_IMPORT_TEST_LIB_DEPS}
  )
