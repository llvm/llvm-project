set(LLVM_LINK_COMPONENTS
  Passes
)

add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY AIIR_EXTENSION_LIBS)
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

aiir_target_link_libraries(tco PRIVATE
  ${dialect_libs}
  ${extension_libs}
  AIIRIR
  AIIRLLVMDialect
  AIIRBuiltinToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
  AIIRTargetLLVMIRExport
  AIIRPass
  AIIRTransforms
  AIIRAffineToStandard
  AIIRAnalysis
  AIIRSCFToControlFlow
  AIIRParser
  AIIRSupport
  AIIRVectorToLLVM
)
