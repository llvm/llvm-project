set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
llvm_tablegen(Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(SrcEditMergeOptsTableGen)

add_clang_tool(clang-ssaf-src-edit-merge
  SrcEditMerge.cpp

  DEPENDS
  SrcEditMergeOptsTableGen
  )

clang_target_link_libraries(clang-ssaf-src-edit-merge
  PRIVATE
  clangAST
  clangToolingCore
  )
