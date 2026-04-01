add_aiir_dialect(SMT smt)

set(LLVM_TARGET_DEFINITIONS SMT.td)

aiir_tablegen(SMTAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(SMTAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRSMTAttrIncGen)

aiir_tablegen(SMTEnums.h.inc -gen-enum-decls)
aiir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRSMTEnumsIncGen)

add_aiir_doc(SMT SMT Dialects/ -gen-dialect-doc)
