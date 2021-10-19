set(LLVM_TARGET_DEFINITIONS CGOps.td)
mlir_tablegen(CGOps.h.inc -gen-op-decls)
mlir_tablegen(CGOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CGOpsIncGen)

set(LLVM_TARGET_DEFINITIONS CGPasses.td)
mlir_tablegen(CGPasses.h.inc -gen-pass-decls -name OptCodeGen)
add_public_tablegen_target(FIROptCodeGenPassIncGen)

