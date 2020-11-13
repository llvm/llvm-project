set(LLVM_LINK_COMPONENTS
  Support)
add_clang_library(clangAPINotes
  APINotesYAMLCompiler.cpp
  LINK_LIBS
    clangBasic)
