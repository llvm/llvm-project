set(LLVM_TARGET_DEFINITIONS ShapeCanonicalization.td)
aiir_tablegen(ShapeCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(AIIRShapeCanonicalizationIncGen)

add_aiir_dialect_library(AIIRShapeDialect
  Shape.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Shape

  DEPENDS
  AIIRShapeOpsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRCallInterfaces
  AIIRCastInterfaces
  AIIRControlFlowInterfaces
  AIIRDialect
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRSideEffectInterfaces
  AIIRTensorDialect
  AIIRUBDialect
  )
