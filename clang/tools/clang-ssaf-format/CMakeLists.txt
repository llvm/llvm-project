set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_tool(clang-ssaf-format
  SSAFFormat.cpp
  )

clang_target_link_libraries(clang-ssaf-format
  PRIVATE
  clangBasic
  clangScalableStaticAnalysisFrameworkCore
  clangScalableStaticAnalysisFrameworkTool
  )

export_executable_symbols_for_plugins(clang-ssaf-format)
