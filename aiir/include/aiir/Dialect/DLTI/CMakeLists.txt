add_subdirectory(TransformOps)

add_aiir_dialect(DLTI dlti)
add_aiir_doc(DLTIAttrs DLTIDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DLTIAttrs.td)
aiir_tablegen(DLTIAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=dlti)
aiir_tablegen(DLTIAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=dlti)
add_aiir_dialect_tablegen_target(AIIRDLTIAttrsIncGen)
