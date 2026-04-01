add_aiir_conversion_library(AIIRShardToMPI
  ShardToMPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ShardToMPI

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRDLTIDialect
  AIIRFuncDialect
  AIIRIR
  AIIRLinalgTransforms
  AIIRAffineDialect
  AIIRArithDialect
  AIIRControlFlowDialect
  AIIRSCFDialect
  AIIRTensorDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRShardDialect
  AIIRMPIDialect
  AIIRTransforms
  )
