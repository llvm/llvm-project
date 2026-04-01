add_aiir_dialect_library(AIIRControlFlowTransforms
  BufferDeallocationOpInterfaceImpl.cpp
  BufferizableOpInterfaceImpl.cpp
  StructuralTypeConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ControlFlow/Transforms

  LINK_LIBS PUBLIC
  AIIRBufferizationDialect
  AIIRBufferizationTransforms
  AIIRControlFlowDialect
  AIIRMemRefDialect
  AIIRIR
)
