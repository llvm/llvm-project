add_clang_tool(ssaf-serialization-format-test
  SSAFSerializationFormatTest.cpp
  ExampleAnalyses.cpp
  )

target_link_libraries(ssaf-serialization-format-test
  PRIVATE
  clangAnalysisScalable
  )
