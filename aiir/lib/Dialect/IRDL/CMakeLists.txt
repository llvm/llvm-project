add_aiir_dialect_library(AIIRIRDL
  IR/IRDL.cpp
  IR/IRDLOps.cpp
  IRDLLoading.cpp
  IRDLSymbols.cpp
  IRDLVerifiers.cpp

  DEPENDS
  AIIRIRDLIncGen
  AIIRIRDLOpsIncGen
  AIIRIRDLTypesIncGen

  LINK_LIBS PUBLIC
  AIIRDialect
  AIIRIR
  AIIRInferTypeOpInterface
  AIIRSideEffectInterfaces
  )
