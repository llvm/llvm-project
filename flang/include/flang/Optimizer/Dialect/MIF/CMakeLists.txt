set(LLVM_TARGET_DEFINITIONS MIFDialect.td)
mlir_tablegen(MIFDialect.h.inc -gen-dialect-decls -dialect=mif)
mlir_tablegen(MIFDialect.cpp.inc -gen-dialect-defs -dialect=mif)

# Add Multi Image Fortran operations
set(LLVM_TARGET_DEFINITIONS MIFOps.td)
mlir_tablegen(MIFOps.h.inc -gen-op-decls)
mlir_tablegen(MIFOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MIFOpsIncGen)
