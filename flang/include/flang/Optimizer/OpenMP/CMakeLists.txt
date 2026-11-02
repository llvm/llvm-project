set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlangOpenMP)

add_flang_tablegen_target(FlangOpenMPPassesIncGen)
