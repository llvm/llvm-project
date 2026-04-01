set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(aiir-text-parser-fuzzer
  aiir-text-parser-fuzzer.cpp
  DUMMY_MAIN DummyParserFuzzer.cpp
)
aiir_target_link_libraries(aiir-text-parser-fuzzer
  PUBLIC
  AIIRIR
  AIIRParser
  AIIRSupport
)

