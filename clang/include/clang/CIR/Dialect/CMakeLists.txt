add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
aiir_tablegen(Passes.h.inc -gen-pass-decls -name CIR)
aiir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix CIR)
aiir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix CIR)
add_public_tablegen_target(AIIRCIRPassIncGen)
