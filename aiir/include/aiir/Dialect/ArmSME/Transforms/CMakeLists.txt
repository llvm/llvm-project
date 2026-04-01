set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSME)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ArmSME)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ArmSME)
aiir_tablegen(PassesEnums.h.inc -gen-enum-decls)
aiir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRArmSMETransformsIncGen)

add_aiir_doc(Passes ArmSMEPasses ./ -gen-pass-doc)
