set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-merge-transformation-report
  MergeTransformationReport.cpp
  )

clang_target_link_libraries(clang-ssaf-merge-transformation-report
  PRIVATE
  clangScalableStaticAnalysisTool
  )
