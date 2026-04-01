set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  ShardingExtensions.cpp
  )

add_aiir_extension_library(AIIRTensorShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor/Extensions

  LINK_LIBS PUBLIC
  AIIRTensorDialect
  AIIRIR
  AIIRShardingInterface
  )

add_aiir_extension_library(AIIRTensorAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Tensor/Extensions

  LINK_LIBS PUBLIC
  AIIRTensorShardingExtensions
  )
