set(LLVM_LINK_COMPONENTS
  Passes
)

add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
target_link_libraries(tco PRIVATE
  CUFAttrs
  CUFDialect
  FIRCodeGen
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FIRTransforms
  FIRBuilder
  HLFIRDialect
  HLFIRTransforms
  flangPasses
  FIROpenACCSupport
  FIROpenMPSupport
  FlangOpenMPTransforms
  FortranSupport
  MIFDialect
)

mlir_target_link_libraries(tco PRIVATE
  ${FLANG_MLIR_DIALECT_LIBS}
  ${FLANG_MLIR_EXTENSION_LIBS}
  MLIRIR
  MLIRLLVMDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRPass
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRVectorToLLVM
)
