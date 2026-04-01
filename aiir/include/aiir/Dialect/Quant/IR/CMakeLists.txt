add_aiir_dialect(QuantOps quant)
add_aiir_doc(QuantOps QuantDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QuantDialectBytecode.td)
aiir_tablegen(QuantDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="Quant")
add_aiir_dialect_tablegen_target(AIIRQuantDialectBytecodeIncGen)
