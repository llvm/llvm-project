set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Tensor)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Tensor)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Tensor)
add_aiir_dialect_tablegen_target(AIIRTensorTransformsIncGen)

add_aiir_doc(Passes TensorPasses ./ -gen-pass-doc)
