set(LLVM_LINK_COMPONENTS
  Core
  Option
  Support
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClangImportTestOptsTableGen)

add_clang_executable(clang-import-test
  clang-import-test.cpp
  DEPENDS
  ClangImportTestOptsTableGen
  intrinsics_gen
  )

set(CLANG_IMPORT_TEST_LIB_DEPS
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangLex
  clangParse
  clangSerialization
  )

clang_target_link_libraries(clang-import-test
  PRIVATE
  ${CLANG_IMPORT_TEST_LIB_DEPS}
  )
