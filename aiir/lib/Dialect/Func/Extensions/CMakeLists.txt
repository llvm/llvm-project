set(LLVM_OPTIONAL_SOURCES
  AllExtensions.cpp
  InlinerExtension.cpp
  ShardingExtensions.cpp
  )

add_aiir_extension_library(AIIRFuncInlinerExtension
  InlinerExtension.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  AIIRControlFlowDialect
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRFuncDialect
  )

add_aiir_extension_library(AIIRFuncShardingExtensions
  ShardingExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRIR
  AIIRShardingInterface
  )


add_aiir_extension_library(AIIRFuncAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/Extensions

  LINK_LIBS PUBLIC
  AIIRFuncInlinerExtension
  AIIRFuncShardingExtensions
  )
