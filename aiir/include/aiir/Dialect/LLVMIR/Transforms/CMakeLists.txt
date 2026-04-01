set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name LLVM)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LLVM)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LLVM)
add_aiir_dialect_tablegen_target(AIIRLLVMPassIncGen)

add_aiir_doc(Passes LLVMPasses ./ -gen-pass-doc)
