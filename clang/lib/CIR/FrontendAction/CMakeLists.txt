set(LLVM_LINK_COMPONENTS
  Core
  FrontendOffloading
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
  MLIRBuiltinLocationAttributesIncGen
  MLIRBytecodeOpInterfaceIncGen
  MLIRBuiltinOpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangCIR
  clangCIRLoweringCommon
  clangCIRLoweringDirectToLLVM
  clangCodeGen
  clangCodeGenUtils
  CIRRegisterAllDialects
  MLIRCIR
  MLIRBytecodeWriter
  MLIRIR
  MLIRParser
  )
