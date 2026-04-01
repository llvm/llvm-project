add_aiir_dialect(StandaloneOps standalone)
add_aiir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_aiir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandalonePasses.td)
aiir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(AIIRStandalonePassesIncGen)
