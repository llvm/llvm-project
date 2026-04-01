add_aiir_library(AIIRShardingInterface
  ShardingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Shard

  DEPENDS
  AIIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRDialectUtils
  AIIRIR
  AIIRShardDialect
  AIIRTensorDialect
  AIIRSupport
)
