add_aiir_dialect_library(AIIRArmSMETransforms
  EnableArmStreaming.cpp
  OuterProductFusion.cpp
  TileAllocation.cpp
  VectorLegalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmSME/Transforms

  DEPENDS
  AIIRArmSMETransformsIncGen

  LINK_LIBS PUBLIC
  AIIRPass
  AIIRArmSMEDialect
  AIIRFuncDialect
  AIIRVectorDialect
  AIIRIndexDialect
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRFuncTransforms
  )
