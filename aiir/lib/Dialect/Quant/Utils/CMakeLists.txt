add_aiir_dialect_library(AIIRQuantUtils
  UniformSupport.cpp
  FakeQuantSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/QuantOps/Utils

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRQuantDialect
  AIIRSupport
  )
