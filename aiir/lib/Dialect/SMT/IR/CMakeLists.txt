add_aiir_dialect_library(AIIRSMT
  SMTAttributes.cpp
  SMTDialect.cpp
  SMTOps.cpp
  SMTTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SMT

  DEPENDS
  AIIRSMTAttrIncGen
  AIIRSMTEnumsIncGen
  AIIRSMTIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  AIIRControlFlowInterfaces
)

add_dependencies(aiir-headers
  AIIRSMTIncGen
)
