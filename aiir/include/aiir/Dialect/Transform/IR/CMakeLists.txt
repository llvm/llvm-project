# Generate the dialect files from the dialect .td.
#
# TODO: Make it possible to use XDialect instead of XOpsDialect in
# add_aiir_dialect.
set(LLVM_TARGET_DEFINITIONS TransformDialect.td)
aiir_tablegen(TransformDialect.h.inc -gen-dialect-decls -dialect=transform)
aiir_tablegen(TransformDialect.cpp.inc -gen-dialect-defs -dialect=transform)
add_aiir_dialect_tablegen_target(AIIRTransformDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TransformTypes.td)
aiir_tablegen(TransformTypes.h.inc -gen-typedef-decls)
aiir_tablegen(TransformTypes.cpp.inc -gen-typedef-defs)
add_aiir_dialect_tablegen_target(AIIRTransformTypesIncGen)
add_aiir_doc(TransformTypes TransformTypes Dialects/ -gen-typedef-doc -dialect=transform)

set(LLVM_TARGET_DEFINITIONS TransformAttrs.td)
aiir_tablegen(TransformDialectEnums.h.inc -gen-enum-decls)
aiir_tablegen(TransformDialectEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectEnumIncGen)
aiir_tablegen(TransformAttrs.h.inc -gen-attrdef-decls)
aiir_tablegen(TransformAttrs.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectAttributesIncGen)

add_aiir_dialect(TransformOps transform)
add_aiir_doc(TransformOps TransformOps Dialects/ -gen-op-doc -dialect=transform)

