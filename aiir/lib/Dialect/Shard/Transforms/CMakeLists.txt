add_aiir_dialect_library(AIIRShardTransforms
  Simplify.cpp
  ShardingPropagation.cpp
  Partition.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Shard

  DEPENDS
  AIIRShardPassIncGen
  AIIRShardingInterface

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineUtils
  AIIRArithDialect
  AIIRArithUtils
  AIIRControlFlowDialect
  AIIRControlFlowInterfaces
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRIR
  AIIRShardDialect
  AIIRPass
  AIIRSupport
  AIIRTensorDialect
  AIIRTosaShardingInterfaceImpl
  AIIRTransformUtils
)
