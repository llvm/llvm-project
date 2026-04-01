add_aiir_dialect(Vector vector)
add_aiir_doc(VectorOps VectorOps Dialects/ -gen-op-doc -dialect=vector)

# Add Vector operations
set(LLVM_TARGET_DEFINITIONS VectorOps.td)
aiir_tablegen(VectorOps.h.inc -gen-op-decls)
aiir_tablegen(VectorOps.cpp.inc -gen-op-defs)
add_aiir_generic_tablegen_target(AIIRVectorOpsIncGen)

# Add Vector attributes
set(LLVM_TARGET_DEFINITIONS VectorAttributes.td)
aiir_tablegen(VectorEnums.h.inc -gen-enum-decls)
aiir_tablegen(VectorEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(VectorAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(VectorAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_generic_tablegen_target(AIIRVectorAttributesIncGen)
