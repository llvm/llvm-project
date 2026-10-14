set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-merge-conflict-report
  MergeConflictReport.cpp
  )

clang_target_link_libraries(clang-ssaf-merge-conflict-report
  PRIVATE
  clangScalableStaticAnalysisTool
  )
