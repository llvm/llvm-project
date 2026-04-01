add_aiir_dialect(XeGPU xegpu)
add_aiir_doc(XeGPU XeGPU Dialects/ -gen-dialect-doc -dialect=xegpu)

set(LLVM_TARGET_DEFINITIONS XeGPU.td)
aiir_tablegen(XeGPUAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=xegpu)
aiir_tablegen(XeGPUAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=xegpu)
add_aiir_dialect_tablegen_target(AIIRXeGPUAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
aiir_tablegen(XeGPUEnums.h.inc -gen-enum-decls)
aiir_tablegen(XeGPUEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRXeGPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
aiir_tablegen(XeGPUAttrInterface.h.inc -gen-attr-interface-decls)
aiir_tablegen(XeGPUAttrInterface.cpp.inc -gen-attr-interface-defs)
add_aiir_dialect_tablegen_target(AIIRXeGPUAttrInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
aiir_tablegen(XeGPUOpInterface.h.inc -gen-op-interface-decls)
aiir_tablegen(XeGPUOpInterface.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRXeGPUOpInterfaceIncGen)
