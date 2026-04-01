add_aiir_conversion_library(AIIRPDLToPDLInterp
  PDLToPDLInterp.cpp
  Predicate.cpp
  PredicateTree.cpp
  RootOrdering.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/PDLToPDLInterp

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRPDLDialect
  AIIRPDLInterpDialect
  AIIRPass
  )
