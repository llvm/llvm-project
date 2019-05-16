add_subdirectory(SampleAnalyzer)
add_subdirectory(CheckerDependencyHandling)
add_subdirectory(CheckerOptionHandling)

set(CLANG_ANALYZER_PLUGIN_DEPS
  SampleAnalyzerPlugin
  CheckerDependencyHandlingAnalyzerPlugin
  CheckerOptionHandlingAnalyzerPlugin
  )

add_custom_target(clang-analyzer-plugin
  DEPENDS ${CLANG_ANALYZER_PLUGIN_DEPS})
