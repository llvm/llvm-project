set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name TargetLLVMIRTransforms)
add_aiir_dialect_tablegen_target(AIIRTargetLLVMIRTransformsIncGen)

add_aiir_doc(Passes TargetLLVMIRTransforms ./ -gen-pass-doc)
