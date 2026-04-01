add_aiir_dialect(PtrOps ptr)
add_aiir_doc(PtrOps PtrOps Dialects/ -gen-dialect-doc -dialect=ptr)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
aiir_tablegen(PtrOpsAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ptr)
aiir_tablegen(PtrOpsAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ptr)
add_aiir_dialect_tablegen_target(AIIRPtrOpsAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS MemorySpaceInterfaces.td)
aiir_tablegen(MemorySpaceAttrInterfaces.h.inc -gen-attr-interface-decls)
aiir_tablegen(MemorySpaceAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_aiir_dialect_tablegen_target(AIIRPtrMemorySpaceInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PtrOps.td)
aiir_tablegen(PtrOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(PtrOpsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRPtrOpsEnumsGen)
