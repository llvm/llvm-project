set(LLVM_TARGET_DEFINITIONS AsyncOps.td)
add_aiir_dialect(AsyncOps async)
add_aiir_doc(AsyncOps AsyncDialect Dialects/ -gen-dialect-doc)
