set(LLVM_TARGET_DEFINITIONS VectorTransformsBase.td)
aiir_tablegen(VectorTransformsEnums.h.inc -gen-enum-decls)
aiir_tablegen(VectorTransformsEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Vector)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Vector)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Vector)
add_aiir_dialect_tablegen_target(AIIRVectorTransformsIncGen)

add_aiir_doc(Passes VectorPasses ./ -gen-pass-doc)
