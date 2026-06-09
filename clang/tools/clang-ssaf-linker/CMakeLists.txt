set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(clang-ssaf-linker
  SSAFLinker.cpp
  )

clang_target_link_libraries(clang-ssaf-linker
  PRIVATE
  clangBasic
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangScalableStaticAnalysisFrameworkTool
  )
