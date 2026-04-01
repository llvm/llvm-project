set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Reducer)
add_aiir_generic_tablegen_target(AIIRReducerIncGen)

set(LLVM_TARGET_DEFINITIONS DialectReductionPatternInterface.td)
aiir_tablegen(DialectReductionPatternInterface.h.inc -gen-dialect-interface-decls)
add_aiir_generic_tablegen_target(AIIRDialectReductionPatternInterfaceIncGen)

add_aiir_doc(Passes ReducerPasses ./ -gen-pass-doc)
