set(LLVM_LINK_COMPONENTS
  BitReader
  Support
  )

add_clang_library(clangAPINotes
  APINotesManager.cpp
  APINotesWriter.cpp
  APINotesReader.cpp
  APINotesYAMLCompiler.cpp
  Types.cpp

  LINK_LIBS
  clangBasic
)
