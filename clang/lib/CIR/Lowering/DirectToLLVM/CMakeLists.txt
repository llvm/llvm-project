set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVM.cpp
  LowerToLLVMIR.cpp
  LowerToLLVMOffloadPrintf.cpp
  LowerToLLVMOffloadPrintfAMDGPU.cpp
  LowerToLLVMOffloadPrintfNVPTX.cpp
  LowerToLLVMOpenCLMetadata.cpp

  DEPENDS
  CIRLowering
  MLIRCIREnumsGen
  MLIRCIRPassIncGen
  mlir-headers

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
  MLIROpenMPToLLVMConversion
  MLIROpenMPToLLVMIRTranslation
  MLIROpenMPTransforms
  MLIRIR
  )

target_include_directories(clangCIRLoweringDirectToLLVM PRIVATE
  ${CLANG_SOURCE_DIR}/lib/CIR/Dialect/Transforms/TargetLowering
  )
