set(LLVM_LINK_COMPONENTS
   core
   native
   Option
   OrcJit
   OrcShared
   OrcTargetProcess
   Support
   Target
   TargetParser
  )

add_clang_library(clangInterpreter
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp

  DEPENDS
  intrinsics_gen
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSema
  clangSerialization
  )
