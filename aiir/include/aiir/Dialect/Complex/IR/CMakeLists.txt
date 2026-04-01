add_aiir_dialect(ComplexOps complex)
add_aiir_doc(ComplexOps ComplexOps Dialects/ -gen-dialect-doc  -dialect=complex)

set(LLVM_TARGET_DEFINITIONS ComplexAttributes.td)
aiir_tablegen(ComplexEnums.h.inc -gen-enum-decls)
aiir_tablegen(ComplexEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(ComplexAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(ComplexAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRComplexAttributesIncGen)
