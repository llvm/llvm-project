add_subdirectory(Attributes)

set(LLVM_TARGET_DEFINITIONS CUFDialect.td)
mlir_tablegen(CUFDialect.h.inc -gen-dialect-decls -dialect=cuf)
mlir_tablegen(CUFDialect.cpp.inc -gen-dialect-defs -dialect=cuf)

set(LLVM_TARGET_DEFINITIONS CUFOps.td)
mlir_tablegen(CUFOps.h.inc -gen-op-decls)
mlir_tablegen(CUFOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(CUFOpsIncGen)
