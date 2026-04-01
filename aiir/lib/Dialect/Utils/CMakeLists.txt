add_aiir_library(AIIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp
  VerificationUtils.cpp

  DEPENDS
  AIIRDialectUtilsIncGen
  AIIRArithDialect

  LINK_LIBS PUBLIC
  AIIRIR
)
