set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIROpInterfacesIncGen
  MLIRBuiltinOpsIncGen

  LINK_LIBS
  clangAST
  clangFrontend
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCodeGen
  MLIRCIR
  MLIRIR
  )
