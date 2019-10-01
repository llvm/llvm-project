set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/SampleAnalyzerPlugin.exports)
add_llvm_library(SampleAnalyzerPlugin MODULE MainCallChecker.cpp PLUGIN_TOOL clang)

target_link_libraries(SampleAnalyzerPlugin PRIVATE
  clangAnalysis
  clangAST
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  LLVMSupport
  )
