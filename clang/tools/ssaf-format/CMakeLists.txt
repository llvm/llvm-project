set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(clang-ssaf-format
  SSAFFormat.cpp
  )

clang_target_link_libraries(clang-ssaf-format
  PRIVATE
  clangScalableStaticAnalysisFrameworkCore
  clangBasic
  )
