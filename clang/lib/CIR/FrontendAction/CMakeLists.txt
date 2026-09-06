set(LLVM_LINK_COMPONENTS
  Core
  ipo
  Linker
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp
  CIRDiagnosticHandler.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIROpInterfacesIncGen
  MLIRBuiltinOpsIncGen

  LINK_LIBS
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
