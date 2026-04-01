# Contrary to what the name claims, this only produces the _op_ interface. The
# type interface is produced manually below.
add_aiir_interface(TransformInterfaces)
add_aiir_doc(TransformInterfaces TransformOpInterfaces Dialects/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS TransformInterfaces.td)
aiir_tablegen(TransformTypeInterfaces.h.inc -gen-type-interface-decls)
aiir_tablegen(TransformTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_aiir_dialect_tablegen_target(AIIRTransformDialectTypeInterfacesIncGen)
add_aiir_doc(TransformInterfaces TransformTypeInterfaces Dialects/ -gen-type-interface-docs)

add_aiir_interface(MatchInterfaces)
add_dependencies(AIIRMatchInterfacesIncGen AIIRTransformInterfacesIncGen)
add_dependencies(aiir-headers AIIRMatchInterfacesIncGen)
add_aiir_doc(MatchInterfaces MatchOpInterfaces Dialects/ -gen-op-interface-docs)

