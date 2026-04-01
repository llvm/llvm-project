set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
aiir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(AIIRSPIRVCanonicalizationIncGen)

add_aiir_dialect_library(AIIRSPIRVDialect
  ArmGraphOps.cpp
  AtomicOps.cpp
  CastOps.cpp
  ControlFlowOps.cpp
  CooperativeMatrixOps.cpp
  DotProductOps.cpp
  GroupOps.cpp
  ImageOps.cpp
  MemoryOps.cpp
  MeshOps.cpp
  SPIRVAttributes.cpp
  SPIRVCanonicalization.cpp
  SPIRVGLCanonicalization.cpp
  SPIRVDialect.cpp
  SPIRVEnums.cpp
  SPIRVOpAvailability.cpp
  SPIRVOpDefinition.cpp
  SPIRVOps.cpp
  SPIRVParsingUtils.cpp
  SPIRVTosaOps.cpp
  SPIRVTypes.cpp
  TargetAndABI.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV

  DEPENDS
  AIIRGPUDialect
  AIIRSPIRVAttributeIncGen
  AIIRSPIRVAttrUtilsGen
  AIIRSPIRVAvailabilityIncGen
  AIIRSPIRVCanonicalizationIncGen
  AIIRSPIRVEnumAvailabilityIncGen
  AIIRSPIRVEnumsIncGen
  AIIRSPIRVImageInterfacesIncGen
  AIIRSPIRVOpsIncGen

  LINK_LIBS PUBLIC
  AIIRControlFlowInterfaces
  AIIRFunctionInterfaces
  AIIRIR
  AIIRParser
  AIIRSideEffectInterfaces
  AIIRSPIRVImageInterfaces
  AIIRSupport
  AIIRTransforms
  AIIRUBDialect
)
