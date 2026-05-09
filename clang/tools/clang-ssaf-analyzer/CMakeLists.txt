set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

if(CLANG_PLUGIN_SUPPORT)
  set(support_plugins SUPPORT_PLUGINS)
endif()

add_clang_tool(clang-ssaf-analyzer
  SSAFAnalyzer.cpp
  ${support_plugins}
  )

clang_target_link_libraries(clang-ssaf-analyzer
  PRIVATE
  clangBasic
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangScalableStaticAnalysisFrameworkTool
  )

if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(clang-ssaf-analyzer)
endif()
