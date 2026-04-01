add_aiir_dialect_library(AIIRMPIDialect
  MPIOps.cpp
  MPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/MPI

  DEPENDS
  AIIRMPIIncGen
  AIIRMPIOpsIncGen
  AIIRMPITypesIncGen
  AIIRMPIAttrsIncGen

  LINK_LIBS PUBLIC
  AIIRDialect
  AIIRDLTIDialect
  AIIRIR
  AIIRMemRefDialect
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  )
