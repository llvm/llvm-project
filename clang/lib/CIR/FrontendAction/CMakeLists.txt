set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBuiltinTypeInterfacesIncGen
  MLIRFunctionInterfacesIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangCodeGen
  clangLex
  clangFrontend
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCIRLoweringThroughMLIR
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  )
