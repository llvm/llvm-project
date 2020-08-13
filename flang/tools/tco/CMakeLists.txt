set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
)

add_llvm_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(tco PRIVATE
  FIROptimizer
  ${dialect_libs}
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRSCFToStandard
  MLIREDSC
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorToLLVM
)
