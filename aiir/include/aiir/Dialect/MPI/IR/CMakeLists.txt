add_aiir_dialect(MPI mpi)
add_aiir_doc(MPIOps MPI Dialects/ -gen-dialect-doc)

# Add MPI operations
set(LLVM_TARGET_DEFINITIONS MPIOps.td)
aiir_tablegen(MPIOps.h.inc -gen-op-decls)
aiir_tablegen(MPIOps.cpp.inc -gen-op-defs)
add_aiir_dialect_tablegen_target(AIIRMPIOpsIncGen)

# Add MPI types
set(LLVM_TARGET_DEFINITIONS MPITypes.td)
aiir_tablegen(MPITypesGen.h.inc -gen-typedef-decls)
aiir_tablegen(MPITypesGen.cpp.inc -gen-typedef-defs)
add_aiir_dialect_tablegen_target(AIIRMPITypesIncGen)

# Add MPI attributes
set(LLVM_TARGET_DEFINITIONS MPI.td)
aiir_tablegen(MPIEnums.h.inc -gen-enum-decls)
aiir_tablegen(MPIEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(MPIAttrDefs.h.inc -gen-attrdef-decls)
aiir_tablegen(MPIAttrDefs.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRMPIAttrsIncGen)
