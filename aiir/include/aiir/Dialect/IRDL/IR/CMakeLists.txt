add_aiir_dialect(IRDL irdl)
add_aiir_doc(IRDLOps IRDLOps Dialects/ -gen-op-doc -dialect=irdl)

# Add IRDL interfaces
set(LLVM_TARGET_DEFINITIONS IRDLInterfaces.td)
aiir_tablegen(IRDLInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(IRDLInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_generic_tablegen_target(AIIRIRDLInterfacesIncGen)

# Add IRDL operations
set(LLVM_TARGET_DEFINITIONS IRDLOps.td)
aiir_tablegen(IRDLOps.h.inc -gen-op-decls)
aiir_tablegen(IRDLOps.cpp.inc -gen-op-defs)
add_aiir_generic_tablegen_target(AIIRIRDLOpsIncGen)

# Add IRDL types
set(LLVM_TARGET_DEFINITIONS IRDLTypes.td)
aiir_tablegen(IRDLTypesGen.h.inc -gen-typedef-decls)
aiir_tablegen(IRDLTypesGen.cpp.inc -gen-typedef-defs)
add_aiir_generic_tablegen_target(AIIRIRDLTypesIncGen)

# Add IRDL attributes
set(LLVM_TARGET_DEFINITIONS IRDLAttributes.td)
aiir_tablegen(IRDLEnums.h.inc -gen-enum-decls)
aiir_tablegen(IRDLEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(IRDLAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(IRDLAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_generic_tablegen_target(AIIRIRDLAttributesIncGen)
