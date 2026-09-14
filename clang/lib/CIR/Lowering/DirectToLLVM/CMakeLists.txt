set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVM.cpp
  LowerToLLVMIR.cpp
  LowerToLLVMOpenCLMetadata.cpp

  DEPENDS
  CIRLowering
  MLIRCIREnumsGen
  MLIRCIROpsIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS
  clangCIRLoweringCommon
  MLIRCIR
  MLIRCIRTargetLowering
  MLIRCIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIROpenMPDialect
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIROpenMPTransforms
  MLIRIR
  )

target_include_directories(clangCIRLoweringDirectToLLVM PRIVATE
  ${CLANG_SOURCE_DIR}/lib/CIR/Dialect/Transforms/TargetLowering
  )
