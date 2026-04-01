add_aiir_dialect(SPIRVOps spirv)
add_aiir_doc(SPIRVOps SPIRVOps Dialects/ -gen-op-doc -dialect=spirv)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
aiir_tablegen(SPIRVEnums.h.inc -gen-enum-decls)
aiir_tablegen(SPIRVEnums.cpp.inc -gen-enum-defs)
add_aiir_dialect_tablegen_target(AIIRSPIRVEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
aiir_tablegen(SPIRVEnumAvailability.h.inc -gen-spirv-enum-avail-decls)
aiir_tablegen(SPIRVEnumAvailability.cpp.inc -gen-spirv-enum-avail-defs)
aiir_tablegen(SPIRVCapabilityImplication.inc -gen-spirv-capability-implication)
add_aiir_dialect_tablegen_target(AIIRSPIRVEnumAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
aiir_tablegen(SPIRVAvailability.h.inc -gen-avail-interface-decls)
aiir_tablegen(SPIRVAvailability.cpp.inc -gen-avail-interface-defs)
aiir_tablegen(SPIRVOpAvailabilityImpl.inc -gen-spirv-avail-impls)
add_aiir_dialect_tablegen_target(AIIRSPIRVAvailabilityIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
aiir_tablegen(SPIRVSerialization.inc -gen-spirv-serialization)
add_aiir_dialect_tablegen_target(AIIRSPIRVSerializationGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
aiir_tablegen(SPIRVAttrUtils.inc -gen-spirv-attr-utils)
add_aiir_dialect_tablegen_target(AIIRSPIRVAttrUtilsGen)

set(LLVM_TARGET_DEFINITIONS SPIRVAttributes.td)
aiir_tablegen(SPIRVAttributes.h.inc -gen-attrdef-decls)
aiir_tablegen(SPIRVAttributes.cpp.inc -gen-attrdef-defs)
add_aiir_dialect_tablegen_target(AIIRSPIRVAttributeIncGen)
