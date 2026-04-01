add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/OpenMP/OMP.td)
aiir_tablegen(OmpCommon.td --gen-directive-decl --directives-dialect=OpenMP)
add_aiir_dialect_tablegen_target(omp_common_td)

set(LLVM_TARGET_DEFINITIONS OpenMPOps.td)

# The OpenMP verifier tablegen pseudo-backend does not produce any output, but
# aiir_tablegen expects an output file name to be passed. An empty "no-output"
# file is created by the statement below.
#
# This output will be added to the list of dependencies of the
# AIIROpenMPOpsIncGen target below, which results in triggering this
# verification pass every time OpenMPOps.td is modified and recompiled.
aiir_tablegen(no-output -verify-openmp-ops)

aiir_tablegen(OpenMPOpsDialect.h.inc -gen-dialect-decls -dialect=omp)
aiir_tablegen(OpenMPOpsDialect.cpp.inc -gen-dialect-defs -dialect=omp)
aiir_tablegen(OpenMPOps.h.inc -gen-op-decls)
aiir_tablegen(OpenMPOps.cpp.inc -gen-op-defs)
aiir_tablegen(OpenMPClauseOps.h.inc -gen-openmp-clause-ops)
aiir_tablegen(OpenMPOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=omp)
aiir_tablegen(OpenMPOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=omp)
aiir_tablegen(OpenMPOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(OpenMPOpsEnums.cpp.inc -gen-enum-defs)
aiir_tablegen(OpenMPOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=omp)
aiir_tablegen(OpenMPOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=omp)
add_aiir_doc(OpenMPOps OpenMPDialect Dialects/ -gen-dialect-doc -dialect=omp)
add_aiir_dialect_tablegen_target(AIIROpenMPOpsIncGen)
add_dependencies(OpenMPDialectDocGen omp_common_td)
add_aiir_interface(OpenMPOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS OpenMPTypeInterfaces.td)
aiir_tablegen(OpenMPTypeInterfaces.h.inc -gen-type-interface-decls)
aiir_tablegen(OpenMPTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_aiir_generic_tablegen_target(AIIROpenMPTypeInterfacesIncGen)
