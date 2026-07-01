set(LLVM_LINK_COMPONENTS
  Option
  Support
  TargetParser
  )

add_clang_tool(clang-ssaf-linker
  SSAFLinker.cpp
  )

clang_target_link_libraries(clang-ssaf-linker
  PRIVATE
  clangBasic
  clangScalableStaticAnalysisAnalyses
  clangScalableStaticAnalysisCore
  clangScalableStaticAnalysisSourceTransformation
  clangScalableStaticAnalysisTool
  )
