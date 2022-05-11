set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
  )

add_llvm_fuzzer(clang-pseudo-fuzzer
  Fuzzer.cpp
  DUMMY_MAIN Main.cpp
  )

target_link_libraries(clang-pseudo-fuzzer
  PRIVATE
  clangPseudo
  )
