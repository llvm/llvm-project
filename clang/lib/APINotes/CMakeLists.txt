set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Support)

add_clang_library(clangAPINotes
  APINotesManager.cpp
  APINotesReader.cpp
  APINotesTypes.cpp
  APINotesWriter.cpp
  APINotesYAMLCompiler.cpp
  Types.cpp

  LINK_LIBS
    clangBasic)
