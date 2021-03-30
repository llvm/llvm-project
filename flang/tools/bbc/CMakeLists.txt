
add_flang_tool(bbc bbc.cpp)
llvm_update_compile_flags(bbc)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(bbc PRIVATE
  FIRDialect
  FIRAnalysis
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  MLIRAffineToStandard
  MLIRSCFToStandard
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
)
