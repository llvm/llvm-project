add_subdirectory(SampleAnalyzer)
add_subdirectory(CheckerDependencyHandling)

set(CLANG_ANALYZER_PLUGIN_DEPS
  SampleAnalyzerPlugin
  CheckerDependencyHandlingAnalyzerPlugin
  )

add_custom_target(clang-analyzer-plugin
  DEPENDS ${CLANG_ANALYZER_PLUGIN_DEPS})
