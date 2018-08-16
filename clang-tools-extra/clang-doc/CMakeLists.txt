set(LLVM_LINK_COMPONENTS
  support
  BitReader
  BitWriter
  )

add_clang_library(clangDoc
  BitcodeReader.cpp
  BitcodeWriter.cpp
  ClangDoc.cpp
  Generators.cpp
  Mapper.cpp
  MDGenerator.cpp
  Representation.cpp
  Serialize.cpp
  YAMLGenerator.cpp

  LINK_LIBS
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
