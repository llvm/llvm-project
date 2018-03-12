set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangDoc
  BitcodeWriter.cpp
  ClangDoc.cpp
  Mapper.cpp
  Serialize.cpp

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
