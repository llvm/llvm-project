add_aiir_dialect_library(AIIRShardDialect
  ShardOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Shard

  DEPENDS
  AIIRShardIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRIR
  AIIRSupport
  AIIRViewLikeInterface
  AIIRDestinationStyleOpInterface
)
