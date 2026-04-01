set(LLVM_TARGET_DEFINITIONS StructuredOpsUtils.td)
aiir_tablegen(DialectUtilsEnums.h.inc -gen-enum-decls)
aiir_tablegen(DialectUtilsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRDialectUtilsIncGen)
