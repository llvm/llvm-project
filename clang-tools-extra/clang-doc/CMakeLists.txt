set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )
add_subdirectory(support)

add_clang_library(clangDoc STATIC
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

  DEPENDS
  omp_gen
  ClangDriverOptions
)

clang_target_link_libraries(clangDoc
  PRIVATE
  clangDocSupport        
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

target_link_libraries(clangDoc
  PRIVATE
  clangDocSupport      
  )

add_subdirectory(tool)
