set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-conflict-merge
  ConflictMerge.cpp
  )

clang_target_link_libraries(clang-ssaf-conflict-merge
  PRIVATE
  clangScalableStaticAnalysisTool
  )
