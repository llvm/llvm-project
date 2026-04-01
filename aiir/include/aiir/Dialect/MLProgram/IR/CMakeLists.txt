set(LLVM_TARGET_DEFINITIONS MLProgramOps.td)
add_aiir_dialect(MLProgramOps ml_program)
add_aiir_doc(MLProgramOps MLProgramOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MLProgramAttributes.td)
aiir_tablegen(MLProgramAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(MLProgramAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRMLProgramAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS MLProgramTypes.td)
aiir_tablegen(MLProgramTypes.h.inc -gen-typedef-decls)
aiir_tablegen(MLProgramTypes.cpp.inc -gen-typedef-defs)
add_aiir_dialect_tablegen_target(AIIRMLProgramTypesIncGen)
