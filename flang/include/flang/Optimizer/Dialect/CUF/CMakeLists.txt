add_subdirectory(Attributes)

set(LLVM_TARGET_DEFINITIONS CUFDialect.td)
aiir_tablegen(CUFDialect.h.inc -gen-dialect-decls -dialect=cuf)
aiir_tablegen(CUFDialect.cpp.inc -gen-dialect-defs -dialect=cuf)

set(LLVM_TARGET_DEFINITIONS CUFOps.td)
aiir_tablegen(CUFOps.h.inc -gen-op-decls)
aiir_tablegen(CUFOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(CUFOpsIncGen)
