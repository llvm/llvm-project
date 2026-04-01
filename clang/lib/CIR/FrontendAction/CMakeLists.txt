set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  DEPENDS
  AIIRCIROpsIncGen
  AIIRCIROpInterfacesIncGen
  AIIRBuiltinOpsIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangCIR
  clangCIRLoweringCommon
  clangCIRLoweringDirectToLLVM
  clangCodeGen
  AIIRCIR
  AIIRIR
  )
