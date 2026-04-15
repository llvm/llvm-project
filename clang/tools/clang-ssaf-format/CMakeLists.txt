set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

if(CLANG_PLUGIN_SUPPORT)
  set(support_plugins SUPPORT_PLUGINS)
endif()

add_clang_tool(clang-ssaf-format
  SSAFFormat.cpp
  ${support_plugins}
  )

clang_target_link_libraries(clang-ssaf-format
  PRIVATE
  clangBasic
  clangScalableStaticAnalysisFrameworkAnalyses
  clangScalableStaticAnalysisFrameworkCore
  clangScalableStaticAnalysisFrameworkTool
  )

if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(clang-ssaf-format)
endif()
