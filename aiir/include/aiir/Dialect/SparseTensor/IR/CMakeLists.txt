add_aiir_dialect(SparseTensorOps sparse_tensor)
add_aiir_doc(SparseTensorOps SparseTensorOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS SparseTensorAttrDefs.td)
aiir_tablegen(SparseTensorAttrDefs.h.inc -gen-attrdef-decls)
aiir_tablegen(SparseTensorAttrDefs.cpp.inc -gen-attrdef-defs)
aiir_tablegen(SparseTensorAttrEnums.h.inc -gen-enum-decls)
aiir_tablegen(SparseTensorAttrEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRSparseTensorAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS SparseTensorTypes.td)
aiir_tablegen(SparseTensorTypes.h.inc -gen-typedef-decls)
aiir_tablegen(SparseTensorTypes.cpp.inc -gen-typedef-defs)
add_aiir_dialect_tablegen_target(AIIRSparseTensorTypesIncGen)

set(LLVM_TARGET_DEFINITIONS SparseTensorInterfaces.td)
aiir_tablegen(SparseTensorInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(SparseTensorInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRSparseTensorInterfacesIncGen)
