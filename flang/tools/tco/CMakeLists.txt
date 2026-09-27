set(LLVM_LINK_COMPONENTS
  Passes
)

add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
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
  ${dialect_libs}
  ${extension_libs}
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
