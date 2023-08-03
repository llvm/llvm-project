set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Support)

add_clang_library(clangAPINotes
  APINotesManager.cpp
  APINotesWriter.cpp
  APINotesReader.cpp
  APINotesTypes.cpp
  APINotesYAMLCompiler.cpp
  Types.cpp

  LINK_LIBS
    clangBasic)
