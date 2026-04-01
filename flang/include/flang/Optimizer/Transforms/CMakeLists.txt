set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name OptTransform)
add_public_tablegen_target(FIROptTransformsPassIncGen)
