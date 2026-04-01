add_aiir_dialect_library(AIIRIndexDialect
  IndexAttrs.cpp
  IndexDialect.cpp
  IndexOps.cpp
  InferIntRangeInterfaceImpls.cpp

  DEPENDS
  AIIRIndexOpsIncGen

  LINK_LIBS PUBLIC
  AIIRDialect
  AIIRIR
  AIIRCastInterfaces
  AIIRInferIntRangeCommon
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  )
