add_flang_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(tco PRIVATE
  FIRCodeGen
  FIRDialect
  FIRSupport
  FIRTransforms
  FIRBuilder
  ${dialect_libs}
  MLIRIR
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToControlFlow
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorToLLVM
)
