set(LLVM_TARGET_DEFINITIONS LinalgMatchOps.td)
aiir_tablegen(LinalgMatchOps.h.inc -gen-op-decls)
aiir_tablegen(LinalgMatchOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgMatchOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
aiir_tablegen(LinalgTransformOps.h.inc -gen-op-decls)
aiir_tablegen(LinalgTransformOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LinalgTransformEnums.td)
aiir_tablegen(LinalgTransformOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(LinalgTransformOpsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRLinalgTransformEnumsIncGen)

add_aiir_doc(LinalgMatchOps LinalgStructuredMatchOps Dialects/ -gen-op-doc -dialect=transform)
add_aiir_doc(LinalgTransformOps LinalgStructuredTransformOps Dialects/ -gen-op-doc -dialect=transform)
