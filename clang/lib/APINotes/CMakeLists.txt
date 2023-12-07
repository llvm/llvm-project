set(LLVM_LINK_COMPONENTS
  Support)
add_clang_library(clangAPINotes
  APINotesTypes.cpp
  APINotesWriter.cpp
  APINotesYAMLCompiler.cpp
  LINK_LIBS
    clangBasic)
