add_aiir_dialect_library(AIIROpenACCUtils
  OpenACCUtils.cpp
  OpenACCUtilsCG.cpp
  OpenACCUtilsGPU.cpp
  OpenACCUtilsLoop.cpp
  OpenACCUtilsTiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenACC

  DEPENDS
  AIIROpenACCPassIncGen
  AIIROpenACCOpsIncGen
  AIIROpenACCEnumsIncGen
  AIIROpenACCAttributesIncGen
  AIIROpenACCMPOpsInterfacesIncGen
  AIIROpenACCOpsInterfacesIncGen
  AIIROpenACCTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRArithUtils
  AIIRDataLayoutInterfaces
  AIIRGPUDialect
  AIIROpenACCDialect
  AIIRIR
  AIIRSCFDialect
  AIIRSCFUtils
  AIIRSupport
  AIIRTransformUtils
)
