add_aiir_dialect(AffineOps affine)
add_aiir_doc(AffineOps AffineOps Dialects/ -gen-op-doc -dialect=affine)

add_aiir_interface(AffineMemoryOpInterfaces)
add_dependencies(AIIRAffineOpsIncGen AIIRAffineMemoryOpInterfacesIncGen)
