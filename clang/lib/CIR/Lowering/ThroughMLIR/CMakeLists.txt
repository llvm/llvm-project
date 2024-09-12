set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringThroughMLIR
  LowerCIRLoopToSCF.cpp
  LowerCIRToMLIR.cpp
  LowerMLIRToLLVM.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
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
  clangCIRLoweringHelpers
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIROpenMPDialect
  MLIROpenMPToLLVMIRTranslation
  )
