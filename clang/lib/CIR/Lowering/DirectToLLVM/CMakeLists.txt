set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVMIR.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRCIREnumsGen
  MLIRCIROpsIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
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
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRCIRTransforms
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIROpenMPDialect
  MLIROpenMPToLLVMIRTranslation
  )
