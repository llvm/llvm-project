add_aiir_dialect(SCFOps scf Ops)
add_aiir_doc(SCFOps SCFDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DeviceMappingInterface.td)
aiir_tablegen(DeviceMappingAttrInterface.h.inc -gen-attr-interface-decls)
aiir_tablegen(DeviceMappingAttrInterface.cpp.inc -gen-attr-interface-defs)
aiir_tablegen(DeviceMappingAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(DeviceMappingAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_generic_tablegen_target(AIIRDeviceMappingInterfacesIncGen)
