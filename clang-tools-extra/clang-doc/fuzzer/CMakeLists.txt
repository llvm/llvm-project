# Resolve "support/Markdown.h" against the parent clang-doc directory.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LLVM_LINK_COMPONENTS
  FuzzerCLI
  Support
  )

# This fuzzer runs on oss-fuzz, so keep it around even if it looks unreferenced.
# With a fuzzing engine configured (LLVM_USE_SANITIZE_COVERAGE or an external
# LLVM_LIB_FUZZING_ENGINE) this builds a real fuzz target; otherwise DUMMY_MAIN
# provides a main() so it still builds and can be replayed over saved inputs.
add_llvm_fuzzer(clang-doc-markdown-fuzzer
  FuzzMarkdown.cpp
  DUMMY_MAIN DummyMarkdownFuzzer.cpp
  )

target_link_libraries(clang-doc-markdown-fuzzer
  PRIVATE
  clangDocSupport
  )
