
set(LLVM_TARGET_DEFINITIONS RewritePatterns.td)
mlir_tablegen(RewritePatterns.inc -gen-rewriters)
add_public_tablegen_target(RewritePatternsIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OptTransform)
add_public_tablegen_target(FIROptTransformsPassIncGen)
