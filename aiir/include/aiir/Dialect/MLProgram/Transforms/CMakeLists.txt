set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name MLProgram)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix MLProgram)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix MLProgram)
add_aiir_dialect_tablegen_target(AIIRMLProgramPassIncGen)

add_aiir_doc(Passes MLProgramPasses ./ -gen-pass-doc)
