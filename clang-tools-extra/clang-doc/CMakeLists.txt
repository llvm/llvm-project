set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )

add_clang_library(clangDoc
  BitcodeReader.cpp
  BitcodeWriter.cpp
  ClangDoc.cpp
  FileHelpersClangDoc.cpp
  Generators.cpp
  HTMLGenerator.cpp
  HTMLMustacheGenerator.cpp      
  Mapper.cpp
  MDGenerator.cpp
  Representation.cpp
  Serialize.cpp
  YAMLGenerator.cpp
        
  DEPENDS
  omp_gen
  ClangDriverOptions
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
