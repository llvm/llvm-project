set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVM.cpp
  LowerToLLVMIR.cpp

  DEPENDS
  AIIRCIREnumsGen
  AIIRCIROpsIncGen
  AIIRCIROpInterfacesIncGen

  LINK_LIBS
  clangCIRLoweringCommon
  ${dialect_libs}
  AIIRCIR
  AIIRCIRTargetLowering
  AIIRBuiltinToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
  AIIROpenMPToLLVMIRTranslation
  AIIRIR
  )

target_include_directories(clangCIRLoweringDirectToLLVM PRIVATE
  ${CLANG_SOURCE_DIR}/lib/CIR/Dialect/Transforms/TargetLowering
  )
