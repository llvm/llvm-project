
set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Conversion)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_aiir_generic_tablegen_target(AIIRConversionPassIncGen)

add_aiir_doc(Passes ConversionPasses ./ -gen-pass-doc)

add_subdirectory(ConvertToLLVM)
