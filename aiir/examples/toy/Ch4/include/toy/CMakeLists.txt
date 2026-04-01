# Most dialects should use add_aiir_dialect().  See examples/standalone.
set(LLVM_TARGET_DEFINITIONS Ops.td)
aiir_tablegen(Ops.h.inc -gen-op-decls)
aiir_tablegen(Ops.cpp.inc -gen-op-defs)
aiir_tablegen(Dialect.h.inc -gen-dialect-decls)
aiir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh4OpsIncGen)

# Most dialects should use add_aiir_interfaces().
set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
aiir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh4ShapeInferenceInterfaceIncGen)
