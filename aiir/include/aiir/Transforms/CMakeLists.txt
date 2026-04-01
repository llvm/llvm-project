
set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
aiir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms)
aiir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms)
add_aiir_dialect_tablegen_target(AIIRTransformsPassIncGen)

set(LLVM_TARGET_DEFINITIONS DialectInlinerInterface.td)
aiir_tablegen(DialectInlinerInterface.h.inc -gen-dialect-interface-decls)
add_aiir_dialect_tablegen_target(AIIRTransformsDialectInterfaceIncGen)

add_aiir_doc(Passes GeneralPasses ./ -gen-pass-doc)
