set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingExtensions.cpp
  )

add_aiir_extension_library(AIIRBufferizationShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization/Extensions

  LINK_LIBS PUBLIC
  AIIRBufferizationDialect
  AIIRIR
  AIIRShardingInterface
  )

add_aiir_extension_library(AIIRBufferizationAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Bufferization/Extensions

  LINK_LIBS PUBLIC
  AIIRBufferizationShardingExtensions
  )
