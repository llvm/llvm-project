set(LLVM_TARGET_DEFINITIONS MIFDialect.td)
aiir_tablegen(MIFDialect.h.inc -gen-dialect-decls -dialect=mif)
aiir_tablegen(MIFDialect.cpp.inc -gen-dialect-defs -dialect=mif)

# Add Multi Image Fortran operations
set(LLVM_TARGET_DEFINITIONS MIFOps.td)
aiir_tablegen(MIFOps.h.inc -gen-op-decls)
aiir_tablegen(MIFOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MIFOpsIncGen)
