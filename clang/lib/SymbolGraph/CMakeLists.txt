set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangSymbolGraph
  API.cpp
  ExtractAPIConsumer.cpp
  DeclarationFragments.cpp
  Serialization.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  )
