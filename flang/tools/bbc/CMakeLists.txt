set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
)
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

add_flang_tool(bbc bbc.cpp)
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
  ${llvm_libs}
)
