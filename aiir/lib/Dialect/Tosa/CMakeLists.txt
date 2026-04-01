add_aiir_dialect_library(AIIRTosaDialect
  IR/TosaOps.cpp
  IR/TosaCanonicalizations.cpp
  IR/TargetEnv.cpp
  Utils/ConversionUtils.cpp
  Utils/QuantUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa

  DEPENDS
  AIIRTosaAttributesIncGen
  AIIRTosaDialectBytecodeIncGen
  AIIRTosaOpsIncGen
  AIIRTosaInterfacesIncGen
  AIIRTosaEnumsIncGen
  AIIRTosaAvailabilityIncGen
  AIIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRDialect
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRQuantUtils
  AIIRSideEffectInterfaces
  AIIRTensorDialect
  AIIRVectorInterfaces
  AIIRViewLikeInterface
  )

add_aiir_dialect_library(AIIRTosaShardingInterfaceImpl
  IR/ShardingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tosa

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRShardDialect
  AIIRShardingInterface
  AIIRSupport
  AIIRTosaDialect
  )

add_subdirectory(Transforms)
