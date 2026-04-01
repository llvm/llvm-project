set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
)
add_llvm_fuzzer(aiir-bytecode-parser-fuzzer
  aiir-bytecode-parser-fuzzer.cpp
  DUMMY_MAIN DummyParserFuzzer.cpp
)
aiir_target_link_libraries(aiir-bytecode-parser-fuzzer
  PUBLIC
  AIIRIR
  AIIRParser
  AIIRSupport
)

