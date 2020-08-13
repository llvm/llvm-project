set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
)

add_llvm_tool(bbc bbc.cpp)
llvm_update_compile_flags(bbc)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(bbc PRIVATE
  FIROptimizer
  ${dialect_libs}
  MLIRLLVMIR
  MLIRAffineToStandard
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
)
