set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSVE)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ArmSVE)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ArmSVE)
add_aiir_dialect_tablegen_target(AIIRArmSVEPassIncGen)

add_aiir_doc(Passes ArmSVEPasses ./ -gen-pass-doc)
