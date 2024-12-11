set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangMove STATIC
  Move.cpp
  HelperDeclRefGraph.cpp

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangMove
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
