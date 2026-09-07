set(LLVM_LINK_COMPONENTS
  Core
  ipo
  Linker
  Support
  )

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp
  CIRDiagnosticHandler.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIROpInterfacesIncGen
  MLIRBuiltinOpsIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic
  clangFrontend
  clangCIR
  clangCIRLoweringCommon
  clangCIRLoweringDirectToLLVM
  clangCodeGen
  MLIRCIR
  MLIRIR
  )
