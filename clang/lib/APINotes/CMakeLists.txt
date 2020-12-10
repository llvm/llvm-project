set(LLVM_LINK_COMPONENTS
  Support)
add_clang_library(clangAPINotes
  APINotesTypes.cpp
  APINotesYAMLCompiler.cpp
  LINK_LIBS
    clangBasic)
