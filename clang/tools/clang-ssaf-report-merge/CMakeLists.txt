set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(clang-ssaf-report-merge
  ReportMerge.cpp
  )

clang_target_link_libraries(clang-ssaf-report-merge
  PRIVATE
  clangScalableStaticAnalysisTool
  )
