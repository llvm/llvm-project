set(LLVM_TARGET_DEFINITIONS Options.td)
llvm_tablegen(Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ClangDriverOptions)
