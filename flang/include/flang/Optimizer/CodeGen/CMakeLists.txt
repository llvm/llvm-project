set(LLVM_TARGET_DEFINITIONS CGPasses.td)
mlir_tablegen(CGPasses.h.inc -gen-pass-decls -name OptCodeGen)
add_flang_tablegen_target(FIROptCodeGenPassIncGen)

