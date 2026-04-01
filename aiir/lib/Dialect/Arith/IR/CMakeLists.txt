set(LLVM_OPTIONAL_SOURCES
  ArithOps.cpp
  ArithDialect.cpp
  InferIntRangeInterfaceImpls.cpp
  ValueBoundsOpInterfaceImpl.cpp
  )

set(LLVM_TARGET_DEFINITIONS ArithCanonicalization.td)
aiir_tablegen(ArithCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(AIIRArithCanonicalizationIncGen)

add_aiir_dialect_library(AIIRArithDialect
  ArithOps.cpp
  ArithDialect.cpp
  InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Arith

  DEPENDS
  AIIRArithOpsIncGen
  AIIRArithOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRCastInterfaces
  AIIRDialect
  AIIRInferIntRangeCommon
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRUBDialect
  )

add_aiir_dialect_library(AIIRArithValueBoundsOpInterfaceImpl
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Arith

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIR
  AIIRValueBoundsOpInterface
  )
