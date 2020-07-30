
set(LLVM_TARGET_DEFINITIONS CGPasses.td)
mlir_tablegen(CGPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(FIROptCodeGenPassIncGen)

add_mlir_doc(Passes -gen-pass-doc OptimizerCodeGenPasses ./)
