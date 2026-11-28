set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClangDiffOptsTableGen)

add_clang_executable(clang-diff
  ClangDiff.cpp

  DEPENDS
  ClangDiffOptsTableGen
  )

clang_target_link_libraries(clang-diff
  PRIVATE
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangToolingASTDiff
  )
