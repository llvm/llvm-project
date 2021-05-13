set(LLVM_LINK_COMPONENTS
   core
   native
   Option
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
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangSema
  clangSerialization
  clangCodeGen
  clangFrontendTool
  )
