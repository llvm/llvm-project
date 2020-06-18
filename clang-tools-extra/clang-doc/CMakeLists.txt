set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )

add_clang_library(clangDoc
  BitcodeReader.cpp
  BitcodeWriter.cpp
  ClangDoc.cpp
  Generators.cpp
  HTMLGenerator.cpp
  Mapper.cpp
  MDGenerator.cpp
  Representation.cpp
  Serialize.cpp
  YAMLGenerator.cpp
)

clang_target_link_libraries(clangDoc
  PRIVATE
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
