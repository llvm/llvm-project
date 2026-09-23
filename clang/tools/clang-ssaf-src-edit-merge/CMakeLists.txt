set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-src-edit-merge
  SrcEditMerge.cpp
  )

clang_target_link_libraries(clang-ssaf-src-edit-merge
  PRIVATE
  clangAST
  clangToolingCore
  )
