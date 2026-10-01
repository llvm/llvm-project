set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FIROpenACC)

add_flang_tablegen_target(FIROpenACCPassesIncGen)
