set(LLVM_LINK_COMPONENTS Support)

add_clang_library(clangSymbolGraphMerger
  SymbolGraphMerger.cpp
  SymbolGraph.cpp
  )

clang_target_link_libraries(clangSymbolGraphMerger
  PRIVATE
  clangBasic
  clangToolingCore
  clangToolingInclusions
  clangExtractAPI
)
