set(LLVM_LINK_COMPONENTS
  Passes
)

add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(tco PRIVATE
  FIRCodeGen
  FIRDialect
  FIRDialectSupport
  FIRSupport
  FIRTransforms
  FIRBuilder
  HLFIRDialect
  HLFIRTransforms
  ${dialect_libs}
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRPass
  MLIRFuncToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRVectorToLLVM
)
