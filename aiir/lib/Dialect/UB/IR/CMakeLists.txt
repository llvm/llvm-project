add_aiir_dialect_library(AIIRUBDialect
  UBOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/UB

  DEPENDS
  AIIRUBOpsIncGen
  AIIRUBOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  )
