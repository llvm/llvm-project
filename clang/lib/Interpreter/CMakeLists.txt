set(LLVM_LINK_COMPONENTS
   core
   native
   OrcJit
   Support
   Target
  )

add_clang_library(clangInterpreter
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangEdit
  clangLex
  clangSema
  clangCodeGen
  clangFrontendTool
  )
