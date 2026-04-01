add_aiir_dialect_library(AIIRQuantDialect
  QuantDialectBytecode.h
  QuantDialectBytecode.cpp
  QuantOps.cpp
  QuantTypes.cpp
  TypeDetail.h
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/QuantOps/IR

  DEPENDS
  AIIRQuantOpsIncGen
  AIIRQuantDialectBytecodeIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  AIIRSupport
  )
