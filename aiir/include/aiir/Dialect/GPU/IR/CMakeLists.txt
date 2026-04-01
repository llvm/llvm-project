add_aiir_dialect(GPUOps gpu)
add_aiir_doc(GPUOps GPUOps Dialects/ -gen-op-doc -dialect=gpu)

set(LLVM_TARGET_DEFINITIONS GPUBase.td)
aiir_tablegen(GPUOpInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(GPUOpInterfaces.cpp.inc -gen-op-interface-defs)
add_aiir_dialect_tablegen_target(AIIRGPUOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ParallelLoopMapperAttr.td)
aiir_tablegen(ParallelLoopMapperEnums.h.inc -gen-enum-decls)
aiir_tablegen(ParallelLoopMapperEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRParallelLoopMapperEnumsGen)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
aiir_tablegen(GPUOpsEnums.h.inc -gen-enum-decls)
aiir_tablegen(GPUOpsEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRGPUOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS CompilationAttrInterfaces.td)
aiir_tablegen(CompilationAttrInterfaces.h.inc -gen-attr-interface-decls)
aiir_tablegen(CompilationAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_aiir_dialect_tablegen_target(AIIRGPUCompilationAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS GPUDeviceMappingAttr.td)
aiir_tablegen(GPUDeviceMapperEnums.h.inc -gen-enum-decls)
aiir_tablegen(GPUDeviceMapperEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRGPUDeviceMapperEnumsGen)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
aiir_tablegen(GPUOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gpu)
aiir_tablegen(GPUOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gpu)
add_aiir_dialect_tablegen_target(AIIRGPUOpsAttributesIncGen)
